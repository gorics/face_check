import json, math, os, random, time
from pathlib import Path
from urllib.parse import urlencode

import requests
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

API = "https://commons.wikimedia.org/w/api.php"
OUT = Path("artifacts")
RAW = Path("_train_images")
OUT.mkdir(exist_ok=True)
RAW.mkdir(exist_ok=True)

PEOPLE = [
    ("taylor_swift", "Taylor Swift"),
    ("zendaya", "Zendaya"),
    ("tom_holland", "Tom Holland"),
]
PER_PERSON = 10
SIZE = 48
LATENT = 24
SEED = 20260929
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

session = requests.Session()
session.headers["User-Agent"] = "CelebrityFaceTrainingSmokeTest/1.0 (GitHub Actions educational experiment)"

ALLOWED = ("cc by", "cc-by", "cc by-sa", "cc-by-sa", "cc0", "public domain", "pd-")


def commons_search(name, limit=50):
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": name, "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|extmetadata|mime|size", "iiurlwidth": 384,
    }
    r = session.get(API, params=params, timeout=40)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    return list(pages.values())


def textmeta(meta, key):
    v = (meta.get(key) or {}).get("value", "")
    return str(v).lower()


def license_ok(meta):
    blob = " ".join([textmeta(meta, "LicenseShortName"), textmeta(meta, "UsageTerms"), textmeta(meta, "LicenseUrl")])
    return any(x in blob for x in ALLOWED), blob


def collect():
    manifest = []
    for ident, name in PEOPLE:
        d = RAW / ident
        d.mkdir(exist_ok=True)
        n = 0
        for page in commons_search(name, 80):
            if n >= PER_PERSON:
                break
            ii = (page.get("imageinfo") or [None])[0]
            if not ii:
                continue
            mime = (ii.get("mime") or "").lower()
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            meta = ii.get("extmetadata") or {}
            ok, lic_blob = license_ok(meta)
            if not ok:
                continue
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            try:
                rr = session.get(url, timeout=40)
                rr.raise_for_status()
                tmp = d / f"{n:02d}.bin"
                tmp.write_bytes(rr.content)
                with Image.open(tmp) as im:
                    im = im.convert("RGB")
                    # conservative center-square crop; no raw image artifact is uploaded
                    side = min(im.size)
                    left = (im.width-side)//2; top=(im.height-side)//2
                    im = im.crop((left, top, left+side, top+side)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
                    jpg = d / f"{n:02d}.jpg"
                    im.save(jpg, quality=92)
                tmp.unlink(missing_ok=True)
                rec = {
                    "identity": ident, "display_name": name,
                    "file_title": page.get("title"), "source_url": ii.get("descriptionurl"),
                    "license": (meta.get("LicenseShortName") or {}).get("value", ""),
                    "license_url": (meta.get("LicenseUrl") or {}).get("value", ""),
                    "artist": (meta.get("Artist") or {}).get("value", ""),
                }
                manifest.append(rec)
                n += 1
                print(f"collected {name}: {n}/{PER_PERSON} :: {page.get('title')}")
            except Exception as e:
                print("skip download", page.get("title"), repr(e))
        if n < 4:
            raise RuntimeError(f"Insufficient license-filtered images for {name}: {n}")
    (OUT / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class FaceDataset(Dataset):
    def __init__(self):
        self.items=[]
        for yi,(ident,name) in enumerate(PEOPLE):
            for p in sorted((RAW/ident).glob("*.jpg")):
                self.items.append((p,yi))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,y=self.items[i]
        im=np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/127.5-1.0
        x=torch.from_numpy(im).permute(2,0,1)
        return x, torch.tensor(y,dtype=torch.long)


class CVAE(nn.Module):
    def __init__(self,nid=len(PEOPLE),latent=LATENT):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(3,32,4,2,1), nn.SiLU(),
            nn.Conv2d(32,64,4,2,1), nn.SiLU(),
            nn.Conv2d(64,128,4,2,1), nn.SiLU(),
        )
        self.mu=nn.Linear(128*6*6,latent)
        self.lv=nn.Linear(128*6*6,latent)
        self.emb=nn.Embedding(nid,16)
        self.fc=nn.Linear(latent+16,128*6*6)
        self.dec=nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.SiLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.SiLU(),
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh(),
        )
    def encode(self,x):
        h=self.enc(x).flatten(1)
        return self.mu(h), self.lv(h)
    def decode(self,z,y):
        h=torch.cat([z,self.emb(y)],1)
        h=self.fc(h).view(-1,128,6,6)
        return self.dec(h)
    def forward(self,x,y):
        mu,lv=self.encode(x)
        std=(0.5*lv).exp(); z=mu+std*torch.randn_like(std)
        return self.decode(z,y),mu,lv


def make_grid(samples, names):
    imgs=[]
    for t in samples:
        a=((t.detach().cpu().clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy()
        imgs.append(Image.fromarray(a))
    w=SIZE*len(imgs); h=SIZE+24
    canvas=Image.new("RGB",(w,h),"white")
    d=ImageDraw.Draw(canvas)
    for i,(im,nm) in enumerate(zip(imgs,names)):
        canvas.paste(im,(i*SIZE,0))
        d.text((i*SIZE+2,SIZE+4),nm[:8],fill="black")
    return canvas


def train():
    ds=FaceDataset()
    dl=DataLoader(ds,batch_size=min(16,len(ds)),shuffle=True,num_workers=0)
    model=CVAE()
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    beta=0.002
    log=[]
    t0=time.time(); steps=0
    for epoch in range(35):
        for x,y in dl:
            recon,mu,lv=model(x,y)
            rec=torch.mean(torch.abs(recon-x))
            kl=-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
            loss=rec+beta*kl
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            steps+=1
            if steps==1 or steps%10==0:
                row={"step":steps,"epoch":epoch+1,"loss":float(loss.detach()),"recon_l1":float(rec.detach()),"kl":float(kl.detach())}
                log.append(row); print(row)
    elapsed=time.time()-t0
    torch.save({
        "state_dict":model.state_dict(),
        "people":PEOPLE,"size":SIZE,"latent":LATENT,"seed":SEED,
        "training":{"steps":steps,"elapsed_seconds":elapsed,"final":log[-1] if log else None}
    }, OUT/"celebrity_cvae.pt")
    (OUT/"training_log.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
    model.eval()
    with torch.no_grad():
        y=torch.arange(len(PEOPLE),dtype=torch.long)
        z=torch.randn(len(PEOPLE),LATENT)
        samples=model.decode(z,y)
    grid=make_grid(samples,[n for _,n in PEOPLE])
    grid.save(OUT/"generated_samples.png")
    summary={
        "device":"cpu","identities":len(PEOPLE),"training_images":len(ds),"steps":steps,
        "parameters":sum(p.numel() for p in model.parameters()),
        "elapsed_seconds":round(elapsed,3),
        "final_loss":log[-1]["loss"] if log else None,
        "note":"Real celebrity photos, license-filtered from Wikimedia Commons. Raw images are not uploaded as artifacts."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    collect()
    train()
