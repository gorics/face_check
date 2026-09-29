import argparse,json,random,shutil,time
from collections import defaultdict
from pathlib import Path
import cv2,gdown,numpy as np,requests,torch,torch.nn as nn,torch.nn.functional as F
from PIL import Image,ImageDraw
from torch.utils.data import Dataset,DataLoader
from gdown.download_folder import _GoogleDriveFile,_parse_embedded_folder_view

ROOT_ID='14V2QCmqjrMXgasbnuZ0NpnLzWU2621fC'; TOTAL_CLASSES=50; CPS=2
TARGET=40; MAXC=70; MINC=20; SIZE=192; LATENT=192; EMBED=48; EPOCHS=40; BATCH=16; SEED=20260929
EXT={'.jpg','.jpeg','.png','.webp','.bmp'}
CASCADE=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))

def sess():
 s=requests.Session();s.headers.update({'User-Agent':'Mozilla/5.0 Chrome/124'});return s

def ls(s,fid):
 r=_parse_embedded_folder_view(sess=s,folder_id=fid,verify=True,timeout=40)
 if r is None: raise RuntimeError('Drive folder listing failed '+fid)
 return r

def classes():
 s=sess();_,root=ls(s,ROOT_ID);tid=None
 for c in root:
  if c[2]==_GoogleDriveFile.TYPE_FOLDER and c[1].strip().lower()=='train': tid=c[0];break
 if not tid: raise RuntimeError('train folder missing')
 _,ch=ls(s,tid);a=[(c[1].zfill(4),c[0]) for c in ch if c[2]==_GoogleDriveFile.TYPE_FOLDER and c[1].isdigit()];a.sort()
 if len(a)<50: raise RuntimeError(f'expected 50 classes, got {len(a)}')
 return a[:50]

def ahash(rgb,side=14):
 g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);sm=cv2.resize(g,(side,side),interpolation=cv2.INTER_AREA);return (sm>sm.mean()).reshape(-1)
def ham(a,b): return int(np.count_nonzero(a!=b))
def crop(rgb):
 H,W=rgb.shape[:2];scale=min(1.0,1200.0/max(H,W));d=cv2.resize(rgb,(int(W*scale),int(H*scale)),interpolation=cv2.INTER_AREA) if scale<1 else rgb
 f=CASCADE.detectMultiScale(cv2.cvtColor(d,cv2.COLOR_RGB2GRAY),scaleFactor=1.055,minNeighbors=3,minSize=(18,18))
 if len(f)==0:return None
 x,y,w,h=max(f,key=lambda b:int(b[2])*int(b[3]))
 if scale<1:x,y,w,h=[int(round(v/scale)) for v in (x,y,w,h)]
 side=int(max(w,h)*2.12);cx,cy=x+w//2,y+h//2;x1,y1=cx-side//2,cy-side//2;x2,y2=x1+side,y1+side
 pl,pt=max(0,-x1),max(0,-y1);pr,pb=max(0,x2-W),max(0,y2-H)
 if any((pl,pt,pr,pb)):
  rgb=cv2.copyMakeBorder(rgb,pt,pb,pl,pr,cv2.BORDER_REFLECT_101);x1+=pl;x2+=pl;y1+=pt;y2+=pt
 c=rgb[y1:y2,x1:x2]
 return None if c.size==0 else cv2.resize(c,(SIZE,SIZE),interpolation=cv2.INTER_AREA)

class DS(Dataset):
 def __init__(self,it,aug=False):self.it=it;self.aug=aug
 def __len__(self):return len(self.it)
 def __getitem__(self,i):
  p,y=self.it[i];a=np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)/127.5-1;x=torch.from_numpy(a).permute(2,0,1)
  if self.aug and torch.rand(())<.5:x=torch.flip(x,[2])
  if self.aug and torch.rand(())<.30:
   gain=0.94+torch.rand(())*.12;x=(x*gain).clamp(-1,1)
  if self.aug and torch.rand(())<.25:x=(x+torch.randn_like(x)*.012).clamp(-1,1)
  return x,torch.tensor(y)

class CVAE(nn.Module):
 def __init__(self,nc):
  super().__init__();self.nc=nc
  self.enc=nn.Sequential(
   nn.Conv2d(3,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),
   nn.Conv2d(32,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),
   nn.Conv2d(64,96,4,2,1),nn.GroupNorm(8,96),nn.SiLU(),
   nn.Conv2d(96,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),
   nn.Conv2d(128,192,4,2,1),nn.GroupNorm(12,192),nn.SiLU(),
   nn.Conv2d(192,256,4,2,1),nn.GroupNorm(16,256),nn.SiLU())
  flat=256*3*3
  self.mu=nn.Linear(flat,LATENT);self.lv=nn.Linear(flat,LATENT)
  self.idh=nn.Sequential(nn.LayerNorm(LATENT),nn.Linear(LATENT,128),nn.SiLU(),nn.Dropout(.10),nn.Linear(128,nc))
  self.emb=nn.Embedding(nc,EMBED);self.fc=nn.Linear(LATENT+EMBED,flat)
  self.dec=nn.Sequential(
   nn.ConvTranspose2d(256,192,4,2,1),nn.GroupNorm(12,192),nn.SiLU(),
   nn.ConvTranspose2d(192,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),
   nn.ConvTranspose2d(128,96,4,2,1),nn.GroupNorm(8,96),nn.SiLU(),
   nn.ConvTranspose2d(96,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),
   nn.ConvTranspose2d(64,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),
   nn.ConvTranspose2d(32,3,4,2,1),nn.Tanh())
 def encode(self,x):h=self.enc(x).flatten(1);return self.mu(h),self.lv(h)
 def decode(self,z,y):return self.dec(self.fc(torch.cat([z,self.emb(y)],1)).view(-1,256,3,3))
 def forward(self,x,y):
  m,l=self.encode(x);z=m+torch.randn_like(m)*torch.exp(.5*l);return self.decode(z,y),m,l,self.idh(m)

def edge_l1(a,b):
 dx1=a[:,:,:,1:]-a[:,:,:,:-1];dx2=b[:,:,:,1:]-b[:,:,:,:-1]
 dy1=a[:,:,1:,:]-a[:,:,:-1,:];dy2=b[:,:,1:,:]-b[:,:,:-1,:]
 return .5*(F.l1_loss(dx1,dx2)+F.l1_loss(dy1,dy2))

def collect(sh,sel,raw,tmp):
 s=sess();counts={};fail=defaultdict(int)
 for ly,(cl,fidfolder) in enumerate(sel):
  od=raw/cl;od.mkdir(parents=True,exist_ok=True);_,ch=ls(s,fidfolder);fs=[(c[0],c[1]) for c in ch if c[2]!=_GoogleDriveFile.TYPE_FOLDER and Path(c[1]).suffix.lower() in EXT]
  random.Random(SEED+sh*100003+ly*1009).shuffle(fs);seen=[];ok=0
  for j,(fid,nm) in enumerate(fs[:MAXC]):
   if ok>=TARGET:break
   fp=tmp/f'{cl}_{j}{Path(nm).suffix.lower() or ".jpg"}'
   try:
    r=gdown.download(id=fid,output=str(fp),quiet=True,use_cookies=False,timeout=35,retries=1)
    if not r or not fp.exists():fail['download']+=1;continue
    with Image.open(fp) as im:rgb=np.asarray(im.convert('RGB'))
    c=crop(rgb)
    if c is None:fail['no_face']+=1;continue
    h=ahash(c)
    if any(ham(h,q)<=9 for q in seen):fail['duplicate']+=1;continue
    seen.append(h);Image.fromarray(c).save(od/f'{ok:02d}.jpg',quality=94);ok+=1
   except Exception as e:fail['exception']+=1;print('skip',sh,cl,nm,type(e).__name__,flush=True)
   finally:fp.unlink(missing_ok=True)
  counts[cl]=ok;print(f'SHARD {sh} {cl}: {ok}/{TARGET} from {len(fs)}',flush=True)
  if ok<MINC:raise RuntimeError(f'{cl} only {ok} usable')
 return counts,dict(fail)

def split(sel,raw,seed):
 tr=[];va=[]
 for y,(cl,_) in enumerate(sel):
  fs=sorted((raw/cl).glob('*.jpg'));random.Random(seed+y*71).shuffle(fs);nv=max(4,round(len(fs)*.2));va += [(p,y) for p in fs[:nv]];tr += [(p,y) for p in fs[nv:]]
 random.Random(seed).shuffle(tr);return tr,va

def evalm(m,dl):
 m.eval();n=0;s=defaultdict(float)
 with torch.no_grad():
  for x,y in dl:
   r,mu,lv,lg=m(x,y);rec=F.l1_loss(r,x);ed=edge_l1(r,x);kl=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp());ce=F.cross_entropy(lg,y);b=x.size(0);n+=b;s['rec']+=float(rec)*b;s['edge']+=float(ed)*b;s['kl']+=float(kl)*b;s['ce']+=float(ce)*b;s['acc']+=float((lg.argmax(1)==y).sum())
 return {k:s[k]/n for k in ('rec','edge','kl','ce','acc')}

def train(sh,sel,raw,out,counts,fail):
 sd=SEED+sh;torch.manual_seed(sd);random.seed(sd);np.random.seed(sd);tr,va=split(sel,raw,sd)
 tl=DataLoader(DS(tr,True),batch_size=min(BATCH,len(tr)),shuffle=True);vl=DataLoader(DS(va),batch_size=min(BATCH,len(va)))
 m=CVAE(len(sel));opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=8e-5);best=None;bs=-9;log=[];t0=time.time()
 for ep in range(1,EPOCHS+1):
  m.train();n=0;ss=defaultdict(float)
  for x,y in tl:
   r,mu,lv,lg=m(x,y);rec=F.l1_loss(r,x);ed=edge_l1(r,x);kl=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp());ce=F.cross_entropy(lg,y)
   loss=rec+.18*ed+.00022*kl+.90*ce
   opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();b=x.size(0);n+=b;ss['loss']+=float(loss.detach())*b;ss['acc']+=float((lg.argmax(1)==y).sum())
  sch.step();v=evalm(m,vl);score=1.15*v['acc']-.20*v['rec']-.05*v['edge'];row={'epoch':ep,'train_loss':ss['loss']/n,'train_acc':ss['acc']/n,'val':v,'score':score};log.append(row);print(json.dumps(row),flush=True)
  if score>bs:bs=score;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 m.load_state_dict(best);v=evalm(m,vl);ids=[x[0] for x in sel]
 sd16={k:(v.half() if torch.is_floating_point(v) else v) for k,v in m.state_dict().items()}
 torch.save({'state_dict':sd16,'global_class_ids':ids,'shard':sh,'resolution':SIZE,'latent':LATENT,'embedding':EMBED,'val':v,'dtype':'fp16-storage'},out/f'koin50_v3_shard_{sh:02d}.pt')
 (out/f'log_{sh:02d}.json').write_text(json.dumps(log,indent=2));m.eval()
 with torch.no_grad():
  y=torch.arange(len(ids));g=torch.Generator().manual_seed(sd+999);z=torch.randn(len(ids),LATENT,generator=g);sm=m.decode(z,y).clamp(-1,1)
 can=Image.new('RGB',(SIZE*len(ids),SIZE+20),'white');dr=ImageDraw.Draw(can)
 for i,cl in enumerate(ids):a=((sm[i]+1)*127.5).byte().permute(1,2,0).numpy();can.paste(Image.fromarray(a),(i*SIZE,0));dr.text((i*SIZE+3,SIZE+3),cl,fill='black')
 can.save(out/f'samples_{sh:02d}.png')
 summ={'shard':sh,'global_class_ids':ids,'accepted_images':sum(counts.values()),'per_class_counts':counts,'failures':fail,'train_images':len(tr),'val_images':len(va),'parameters':sum(p.numel() for p in m.parameters()),'epochs':EPOCHS,'training_seconds':round(time.time()-t0,3),'val_identity_accuracy':v['acc'],'chance_identity_accuracy':1/len(ids),'val_reconstruction_l1':v['rec'],'val_edge_l1':v['edge'],'resolution':SIZE,'latent':LATENT,'storage_dtype':'fp16','raw_photos_uploaded':False,'usage_scope':'KoIn official README: academic purposes'}
 (out/f'summary_{sh:02d}.json').write_text(json.dumps(summ,indent=2));print('FINAL',json.dumps(summ),flush=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--shard',type=int,required=True);q=a.parse_args();sh=q.shard
 if not 0<=sh<25:raise SystemExit('shard 0..24')
 allc=classes();sel=allc[sh*CPS:(sh+1)*CPS];out=Path(f'artifacts_koin50_v3_{sh:02d}');raw=Path(f'_crops_v3_{sh:02d}');tmp=Path(f'_tmp_v3_{sh:02d}');out.mkdir(exist_ok=True);raw.mkdir(exist_ok=True);tmp.mkdir(exist_ok=True)
 try:c,f=collect(sh,sel,raw,tmp);train(sh,sel,raw,out,c,f)
 finally:shutil.rmtree(raw,ignore_errors=True);shutil.rmtree(tmp,ignore_errors=True)
if __name__=='__main__':main()
