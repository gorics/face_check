import hashlib
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

API = "https://commons.wikimedia.org/w/api.php"
OUT = Path("artifacts_v2")
RAW = Path("_train_images_v2")
OUT.mkdir(exist_ok=True)
RAW.mkdir(exist_ok=True)

SIZE = 64
LATENT = 64
EMBED = 32
SEED = 20260929
EPOCHS = 70
BATCH = 16
MIN_PER_ID = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

PEOPLE = [
    ("taylor_swift", "Taylor Swift"),
    ("zendaya", "Zendaya"),
    ("tom_holland", "Tom Holland"),
    ("emma_watson", "Emma Watson"),
    ("margot_robbie", "Margot Robbie"),
    ("chris_hemsworth", "Chris Hemsworth"),
    ("ryan_gosling", "Ryan Gosling"),
]

FILES = {
    "taylor_swift": [
        "File:Swift, Taylor (2007).jpg",
        "File:Swift, Taylor (2007) cropped.jpg",
        "File:Swift, Taylor (2007) cropped 2.JPG",
        "File:Taylor Swift at Yahoo 2007.jpg",
        "File:191125 Taylor Swift at the 2019 American Music Awards.png",
        "File:191125 Taylor Swift at the 2019 American Music Awards (2).png",
        "File:191125 Taylor Swift at the 2019 American Music Awards (cropped).png",
        "File:Taylor Swift 2019 by Glenn Francis.jpg",
    ],
    "zendaya": [
        "File:Zendaya - 2019 by Glenn Francis.jpg",
        "File:Zendaya 2019 by Glenn Francis (cropped).jpg",
        "File:Zendaya 2019 by Glenn Francis.jpg",
        "File:Zendaya Full Length - 2019 by Glenn Francis (cropped).jpg",
        "File:Zendaya Full Length - 2019 by Glenn Francis.jpg",
        "File:Zendaya 2024.jpg",
        "File:Zendaya 2026 (cropped).jpg",
        "File:Zendaya 2026.jpg",
    ],
    "tom_holland": [
        "File:Tom Holland (28035716544).jpg",
        "File:Tom Holland (28036487013).jpg",
        "File:Tom Holland (28620384206).jpg",
        "File:Tom Holland (28652884235) (cropped).jpg",
        "File:Tom Holland (28652888235).jpg",
        "File:Tom Holland (28652891355) (cropped).jpg",
        "File:Tom Holland by Gage Skidmore.jpg",
        "File:Tom Holland MTV 2018 (01).jpg",
    ],
    "emma_watson": [
        "File:Emma Watson at Harry Potter and the Half-Blood Prince Premiere 01.jpg",
        "File:Emma Watson at Harry Potter and the Half-Blood Prince Premiere 02.jpg",
        "File:Emma Watson at Harry Potter and the Half-Blood Prince Premiere 06 cropped.jpg",
        "File:Emma Watson at Harry Potter and the Half-Blood Prince Premiere 07.jpg",
        "File:Emma Watson 2010 2.jpg",
        "File:Emma Watson 2010.jpg",
        "File:Emma Watson, November 2010.jpg",
        "File:Emma Watson 2012 Shankbone (cropped).JPG",
    ],
    "margot_robbie": [
        "File:Margot Robbie (cropped).jpg",
        "File:Margot Robbie at Somerset House in 2013 (cropped).jpg",
        "File:Margot Robbie (28129125629) (cropped).jpg",
        "File:Margot Robbie 2018 (cropped).png",
        "File:Margot Robbie MTV 2018 (cropped).png",
        "File:29th Critics Choice Awards - Margot Robbie (cropped 2).png",
        "File:29th Critics Choice Awards - Margot Robbie (cropped).png",
        "File:Margot Robbie at 29th Critics' Choice Awards.jpg",
    ],
    "chris_hemsworth": [
        "File:Chris Hemsworth (7400856286).jpg",
        "File:Chris Hemsworth (7400857106).jpg",
        "File:Chris Hemsworth (7400858240).jpg",
        "File:Chris Hemsworth (7400859048).jpg",
        "File:Chris Hemsworth 3, 2012 (cropped).jpg",
        "File:Hemsworth TFF (cropped).jpg",
        "File:Chris Hemsworth by Gage Skidmore.jpg",
        "File:Chris Hemsworth by Gage Skidmore 3.jpg",
    ],
    "ryan_gosling": [
        "File:Ryan Gosling (35397111013) (cropped 2).jpg",
        "File:Ryan Gosling (35397134013) (cropped).jpg",
        "File:Ryan Gosling (36034827222) (cropped).jpg",
        "File:Ryan Gosling 2017 by Gage Skidmore.jpg",
        "File:Ryan Gosling 2017 crop.jpg",
        "File:Ryan Gosling at SSIFF 2018 (1) (cropped).jpg",
        "File:Ryan Gosling at SSIFF 2018 (2) (cropped).jpg",
        "File:Ryan Gosling in 2018 croped.jpg",
    ],
}

session = requests.Session()
session.headers["User-Agent"] = "CelebrityFaceCVAE-v2/1.0 (GitHub Actions; license-aware Wikimedia Commons experiment)"
ALLOWED = ("cc by", "cc-by", "cc by-sa", "cc-by-sa", "cc0", "public domain", "pd-")
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def retry_get(url, *, params=None, attempts=7, timeout=45):
    last = None
    for k in range(attempts):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = min(12.0, 1.0 + 1.8 * (k + 1))
                print(f"429 retry in {wait:.1f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(min(8.0, 0.7 * (k + 1)))
    if last is None:
        raise RuntimeError("request failed without exception")
    raise last


def api_get(params):
    p = dict(params)
    p["format"] = "json"
    return retry_get(API, params=p).json()


def image_info(title):
    data = api_get({
        "action": "query",
        "prop": "imageinfo",
        "titles": title,
        "iiprop": "url|extmetadata|mime|size|sha1",
        "iiurlwidth": 768,
    })
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    return (page.get("imageinfo") or [None])[0]


def textmeta(meta, key):
    return str((meta.get(key) or {}).get("value", "")).lower()


def license_ok(meta):
    blob = " ".join([
        textmeta(meta, "LicenseShortName"),
        textmeta(meta, "UsageTerms"),
        textmeta(meta, "LicenseUrl"),
    ])
    return any(x in blob for x in ALLOWED)


def ahash(rgb, side=16):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA)
    bits = (g >= g.mean()).reshape(-1)
    return bits


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def face_crop(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    H, W = rgb.shape[:2]
    face_ratio = (w * h) / float(max(1, W * H))
    if face_ratio < 0.015:
        return None, None
    side = int(max(w, h) * 1.75)
    cx, cy = x + w // 2, y + h // 2
    x1, y1 = cx - side // 2, cy - side // 2
    x2, y2 = x1 + side, y1 + side
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - W), max(0, y2 - H)
    if any((pad_l, pad_t, pad_r, pad_b)):
        rgb = cv2.copyMakeBorder(rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
        x1 += pad_l; x2 += pad_l; y1 += pad_t; y2 += pad_t
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    return crop, {"face_ratio": round(face_ratio, 5), "sharpness": round(sharpness, 3)}


def collect():
    manifest = []
    counts = {}
    for ident, display_name in PEOPLE:
        d = RAW / ident
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        hashes = []
        for title in FILES[ident]:
            try:
                ii = image_info(title)
                if not ii:
                    print("missing", title)
                    continue
                mime = (ii.get("mime") or "").lower()
                if mime not in {"image/jpeg", "image/png", "image/webp"}:
                    print("mime rejected", title, mime)
                    continue
                meta = ii.get("extmetadata") or {}
                if not license_ok(meta):
                    print("license rejected", title)
                    continue
                url = ii.get("thumburl") or ii.get("url")
                if not url:
                    continue
                content = retry_get(url).content
                tmp = d / "_tmp_image"
                tmp.write_bytes(content)
                with Image.open(tmp) as im:
                    rgb = np.asarray(im.convert("RGB"))
                tmp.unlink(missing_ok=True)
                crop, quality = face_crop(rgb)
                if crop is None:
                    print("no usable face", title)
                    continue
                h = ahash(crop)
                if any(hamming(h, prev) <= 5 for prev in hashes):
                    print("near duplicate", title)
                    continue
                hashes.append(h)
                path = d / f"{n:02d}.jpg"
                Image.fromarray(crop).save(path, quality=95)
                manifest.append({
                    "identity": ident,
                    "display_name": display_name,
                    "file_title": title,
                    "source_url": ii.get("descriptionurl"),
                    "license": (meta.get("LicenseShortName") or {}).get("value", ""),
                    "license_url": (meta.get("LicenseUrl") or {}).get("value", ""),
                    "artist": (meta.get("Artist") or {}).get("value", ""),
                    "commons_sha1": ii.get("sha1"),
                    "selection": "explicit person-specific Wikimedia Commons file list",
                    "crop": "largest detected frontal face with context margin",
                    "quality": quality,
                })
                n += 1
                print(f"accepted {display_name}: {n} :: {title}")
                time.sleep(0.12)
            except Exception as e:
                print("skip", title, repr(e))
        counts[ident] = n
        if n < MIN_PER_ID:
            raise RuntimeError(f"Insufficient verified face crops for {display_name}: {n}")
    (OUT / "sources_v2.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "counts_v2.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


class FaceDataset(Dataset):
    def __init__(self, augment=True):
        self.augment = augment
        self.items = []
        for yi, (ident, _) in enumerate(PEOPLE):
            for p in sorted((RAW / ident).glob("*.jpg")):
                self.items.append((p, yi))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        if self.augment:
            if random.random() < 0.5:
                a = a[:, ::-1].copy()
            gain = random.uniform(0.92, 1.08)
            bias = random.uniform(-0.035, 0.035)
            a = np.clip(a * gain + bias, 0.0, 1.0)
        x = torch.from_numpy(a * 2.0 - 1.0).permute(2, 0, 1).float()
        return x, torch.tensor(y, dtype=torch.long)


class CVAEv2(nn.Module):
    def __init__(self, nid=len(PEOPLE)):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 192, 4, 2, 1), nn.GroupNorm(12, 192), nn.SiLU(),
        )
        flat = 192 * 4 * 4
        self.mu = nn.Linear(flat, LATENT)
        self.lv = nn.Linear(flat, LATENT)
        self.identity_head = nn.Sequential(nn.Linear(LATENT, 96), nn.SiLU(), nn.Linear(96, nid))
        self.emb = nn.Embedding(nid, EMBED)
        self.fc = nn.Linear(LATENT + EMBED, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(192, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.mu(h), self.lv(h)

    def decode(self, z, y):
        h = torch.cat([z, self.emb(y)], dim=1)
        h = self.fc(h).view(-1, 192, 4, 4)
        return self.dec(h)

    def forward(self, x, y):
        mu, lv = self.encode(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z, y)
        logits = self.identity_head(mu)
        return recon, mu, lv, logits


def save_grid(model, path):
    model.eval()
    with torch.no_grad():
        rows = []
        for seed_offset in (0, 1):
            g = torch.Generator().manual_seed(SEED + 100 + seed_offset)
            y = torch.arange(len(PEOPLE), dtype=torch.long)
            z = torch.randn(len(PEOPLE), LATENT, generator=g)
            rows.append(model.decode(z, y))
        samples = torch.cat(rows, dim=0)
    cols = len(PEOPLE)
    canvas = Image.new("RGB", (SIZE * cols, (SIZE + 20) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    names = [n for _, n in PEOPLE]
    for idx, t in enumerate(samples):
        r, c = divmod(idx, cols)
        arr = ((t.detach().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        canvas.paste(Image.fromarray(arr), (c * SIZE, r * (SIZE + 20)))
        draw.text((c * SIZE + 2, r * (SIZE + 20) + SIZE + 2), names[c][:10], fill="black")
    canvas.save(path)


def train(counts):
    ds = FaceDataset(augment=True)
    dl = DataLoader(ds, batch_size=min(BATCH, len(ds)), shuffle=True, num_workers=0, drop_last=False)
    model = CVAEv2()
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=2e-4)
    log = []
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    global_step = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        sums = {"loss": 0.0, "rec": 0.0, "kl": 0.0, "ce": 0.0, "acc": 0.0}
        nb = 0
        for x, y in dl:
            recon, mu, lv, logits = model(x, y)
            rec = F.l1_loss(recon, x)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            ce = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean()
            global_step += 1
            beta = 0.003 * min(1.0, global_step / 60.0)
            loss = rec + beta * kl + 0.08 * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sums["loss"] += float(loss.detach())
            sums["rec"] += float(rec.detach())
            sums["kl"] += float(kl.detach())
            sums["ce"] += float(ce.detach())
            sums["acc"] += float(acc.detach())
            nb += 1
        sched.step()
        row = {
            "epoch": epoch,
            "step": global_step,
            "loss": sums["loss"] / nb,
            "recon_l1": sums["rec"] / nb,
            "kl": sums["kl"] / nb,
            "identity_ce": sums["ce"] / nb,
            "identity_acc": sums["acc"] / nb,
            "lr": opt.param_groups[0]["lr"],
        }
        log.append(row)
        if row["loss"] < best["loss"]:
            best["loss"] = row["loss"]
            best["epoch"] = epoch
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(row))

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    elapsed = time.time() - t0
    checkpoint = {
        "format": "celebrity-cvae-v2",
        "state_dict": model.state_dict(),
        "people": PEOPLE,
        "size": SIZE,
        "latent": LATENT,
        "embedding": EMBED,
        "seed": SEED,
        "counts": counts,
        "training": {
            "epochs": EPOCHS,
            "steps": global_step,
            "elapsed_seconds": elapsed,
            "best_epoch": best["epoch"],
            "best_loss": best["loss"],
            "final": log[-1],
        },
    }
    torch.save(checkpoint, OUT / "celebrity_cvae_v2.pt")
    (OUT / "training_log_v2.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    save_grid(model, OUT / "generated_samples_v2.png")

    summary = {
        "device": "cpu",
        "identities": len(PEOPLE),
        "training_images": len(ds),
        "per_identity": counts,
        "resolution": SIZE,
        "latent": LATENT,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": EPOCHS,
        "steps": global_step,
        "elapsed_seconds": round(elapsed, 3),
        "best_epoch": best["epoch"],
        "best_loss": best["loss"],
        "final_identity_acc": log[-1]["identity_acc"],
        "data_quality": "explicit person-specific Wikimedia Commons files; free-license filter; face detection; near-duplicate filter",
        "raw_training_images_uploaded": False,
    }
    (OUT / "summary_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    counts = collect()
    train(counts)
