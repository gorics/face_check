import hashlib, json, random, time
from pathlib import Path

import cv2
import requests
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

API = "https://commons.wikimedia.org/w/api.php"
OUT = Path("artifacts")
RAW = Path("_train_images")
OUT.mkdir(exist_ok=True)
RAW.mkdir(exist_ok=True)

PEOPLE = [
    ("taylor_swift", "Taylor Swift", ["Taylor Swift by year"]),
    ("zendaya", "Zendaya", ["Zendaya by year"]),
    ("tom_holland", "Tom Holland", ["Tom Holland (actor) by year", "Tom Holland (actor) at Comic-Con International"]),
]
PER_PERSON = 12
SIZE = 48
LATENT = 24
SEED = 20260929
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

session = requests.Session()
session.headers["User-Agent"] = "CelebrityFaceTrainingSmokeTest/2.0 (GitHub Actions; license-aware Wikimedia Commons experiment)"
ALLOWED = ("cc by", "cc-by", "cc by-sa", "cc-by-sa", "cc0", "public domain", "pd-")
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def api_get(params, attempts=5):
    params = dict(params)
    params["format"] = "json"
    last = None
    for k in range(attempts):
        try:
            r = session.get(API, params=params, timeout=40)
            if r.status_code == 429:
                time.sleep(1.5 * (k + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.5 * (k + 1))
    raise last


def category_members(category):
    cont = None
    while True:
        p = {
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmtype": "file|subcat", "cmlimit": 500,
        }
        if cont:
            p["cmcontinue"] = cont
        data = api_get(p)
        for m in data.get("query", {}).get("categorymembers", []):
            yield m
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break


def crawl_categories(roots, max_depth=3):
    queue = [(r, 0) for r in roots]
    seen_cats, seen_files = set(), set()
    files = []
    while queue:
        cat, depth = queue.pop(0)
        if cat in seen_cats or depth > max_depth:
            continue
        seen_cats.add(cat)
        print(f"crawl category depth={depth}: {cat}")
        try:
            for m in category_members(cat):
                title = m.get("title", "")
                ns = m.get("ns")
                if ns == 14 and depth < max_depth:
                    sub = title.removeprefix("Category:")
                    if sub not in seen_cats:
                        queue.append((sub, depth + 1))
                elif ns == 6 and title not in seen_files:
                    seen_files.add(title)
                    files.append((title, cat))
        except Exception as e:
            print("category skip", cat, repr(e))
    return files


def image_info(title):
    data = api_get({
        "action": "query", "prop": "imageinfo", "titles": title,
        "iiprop": "url|extmetadata|mime|size", "iiurlwidth": 512,
    })
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    return (page.get("imageinfo") or [None])[0]


def textmeta(meta, key):
    return str((meta.get(key) or {}).get("value", "")).lower()


def license_ok(meta):
    blob = " ".join([textmeta(meta, "LicenseShortName"), textmeta(meta, "UsageTerms"), textmeta(meta, "LicenseUrl")])
    return any(x in blob for x in ALLOWED), blob


def download(url, attempts=5):
    last = None
    for k in range(attempts):
        try:
            r = session.get(url, timeout=45)
            if r.status_code == 429:
                time.sleep(1.5 * (k + 1))
                continue
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(0.6 * (k + 1))
    raise last


def largest_face_crop(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    H, W = rgb.shape[:2]
    side = int(max(w, h) * 1.9)
    cx, cy = x + w // 2, y + h // 2
    x1 = max(0, cx - side // 2); x2 = min(W, cx + side // 2)
    y1 = max(0, cy - side // 2); y2 = min(H, cy + side // 2)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return crop


def collect():
    manifest = []
    for ident, name, roots in PEOPLE:
        d = RAW / ident
        d.mkdir(exist_ok=True)
        candidates = crawl_categories(roots, max_depth=3)
        random.shuffle(candidates)
        n = 0
        pixel_hashes = set()
        for title, source_cat in candidates:
            if n >= PER_PERSON:
                break
            try:
                ii = image_info(title)
                if not ii:
                    continue
                mime = (ii.get("mime") or "").lower()
                if mime not in {"image/jpeg", "image/png", "image/webp"}:
                    continue
                meta = ii.get("extmetadata") or {}
                ok, _ = license_ok(meta)
                if not ok:
                    continue
                url = ii.get("thumburl") or ii.get("url")
                if not url:
                    continue
                content = download(url)
                tmp = d / "_tmp_image"
                tmp.write_bytes(content)
                with Image.open(tmp) as im:
                    rgb = np.asarray(im.convert("RGB"))
                tmp.unlink(missing_ok=True)
                crop = largest_face_crop(rgb)
                if crop is None:
                    print("no face:", title)
                    continue
                digest = hashlib.sha256(crop.tobytes()).hexdigest()
                if digest in pixel_hashes:
                    continue
                pixel_hashes.add(digest)
                jpg = d / f"{n:02d}.jpg"
                Image.fromarray(crop).save(jpg, quality=94)
                rec = {
                    "identity": ident, "display_name": name,
                    "source_category": source_cat, "file_title": title,
                    "source_url": ii.get("descriptionurl"),
                    "license": (meta.get("LicenseShortName") or {}).get("value", ""),
                    "license_url": (meta.get("LicenseUrl") or {}).get("value", ""),
                    "artist": (meta.get("Artist") or {}).get("value", ""),
                    "face_crop": "largest detected frontal face",
                }
                manifest.append(rec)
                n += 1
                print(f"accepted {name}: {n}/{PER_PERSON} :: {title}")
                time.sleep(0.12)
            except Exception as e:
                print("skip", title, repr(e))
        if n < 6:
            raise RuntimeError(f"Insufficient verified face crops for {name}: {n}")
    (OUT / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class FaceDataset(Dataset):
    def __init__(self):
        self.items = []
        for yi, (ident, name, roots) in enumerate(PEOPLE):
            for p in sorted((RAW / ident).glob("*.jpg")):
                self.items.append((p, yi))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(im).permute(2, 0, 1)
        return x, torch.tensor(y, dtype=torch.long)


class CVAE(nn.Module):
    def __init__(self, nid=len(PEOPLE), latent=LATENT):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
        )
        self.mu = nn.Linear(128 * 6 * 6, latent)
        self.lv = nn.Linear(128 * 6 * 6, latent)
        self.emb = nn.Embedding(nid, 16)
        self.fc = nn.Linear(latent + 16, 128 * 6 * 6)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )
    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.mu(h), self.lv(h)
    def decode(self, z, y):
        h = torch.cat([z, self.emb(y)], 1)
        return self.dec(self.fc(h).view(-1, 128, 6, 6))
    def forward(self, x, y):
        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return self.decode(z, y), mu, lv


def make_grid(samples, names):
    imgs = []
    for t in samples:
        a = ((t.detach().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
        imgs.append(Image.fromarray(a))
    canvas = Image.new("RGB", (SIZE * len(imgs), SIZE + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (im, name) in enumerate(zip(imgs, names)):
        canvas.paste(im, (i * SIZE, 0))
        draw.text((i * SIZE + 2, SIZE + 4), name[:8], fill="black")
    return canvas


def train():
    ds = FaceDataset()
    dl = DataLoader(ds, batch_size=min(12, len(ds)), shuffle=True, num_workers=0)
    model = CVAE()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    beta = 0.002
    log = []
    t0 = time.time(); steps = 0
    for epoch in range(50):
        for x, y in dl:
            recon, mu, lv = model(x, y)
            rec = torch.mean(torch.abs(recon - x))
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = rec + beta * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            if steps == 1 or steps % 10 == 0:
                row = {"step": steps, "epoch": epoch + 1, "loss": float(loss.detach()), "recon_l1": float(rec.detach()), "kl": float(kl.detach())}
                log.append(row); print(row)
    elapsed = time.time() - t0
    torch.save({
        "state_dict": model.state_dict(),
        "people": [(i, n) for i, n, _ in PEOPLE],
        "size": SIZE, "latent": LATENT, "seed": SEED,
        "training": {"steps": steps, "elapsed_seconds": elapsed, "final": log[-1] if log else None},
    }, OUT / "celebrity_cvae.pt")
    (OUT / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    model.eval()
    with torch.no_grad():
        y = torch.arange(len(PEOPLE), dtype=torch.long)
        z = torch.randn(len(PEOPLE), LATENT)
        samples = model.decode(z, y)
    make_grid(samples, [n for _, n, _ in PEOPLE]).save(OUT / "generated_samples.png")
    summary = {
        "device": "cpu", "identities": len(PEOPLE), "training_images": len(ds), "steps": steps,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": round(elapsed, 3),
        "final_loss": log[-1]["loss"] if log else None,
        "data_quality": "Wikimedia Commons person-specific year categories + allowed-license filter + largest-face crop",
        "note": "Raw third-party training images are intentionally not uploaded as artifacts."
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    collect()
    train()
