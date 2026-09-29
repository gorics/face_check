import json
import math
import os
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

# Official KoIn50 folder linked by dukong1/KoIn_Benchmark_Dataset.
KOIN50_ROOT_ID = "14V2QCmqjrMXgasbnuZ0NpnLzWU2621fC"
OUT = Path("artifacts_koin50")
CROPS = Path("_koin50_crops")
TMP = Path("_koin50_tmp")
OUT.mkdir(exist_ok=True)
CROPS.mkdir(exist_ok=True)
TMP.mkdir(exist_ok=True)

SEED = 20260929
SIZE = 64
LATENT = 128
EMBED = 64
TARGET_PER_CLASS = 12
MAX_CANDIDATES_PER_CLASS = 30
MIN_PER_CLASS = 6
EPOCHS = 32
BATCH = 32
NUM_CLASSES = 50

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
    })
    return s


def list_folder(sess, fid):
    result = _parse_embedded_folder_view(sess=sess, folder_id=fid, verify=True, timeout=45)
    if result is None:
        raise RuntimeError(f"Cannot list folder {fid}")
    return result


def discover_train_classes():
    s = session()
    root_name, root_children = list_folder(s, KOIN50_ROOT_ID)
    train = None
    for cid, cname, ctype, *_ in root_children:
        if ctype == _GoogleDriveFile.TYPE_FOLDER and cname.strip().lower() == "train":
            train = (cid, cname)
            break
    if train is None:
        raise RuntimeError("KoIn50/train folder not found")

    _, children = list_folder(s, train[0])
    classes = []
    for cid, cname, ctype, *_ in children:
        if ctype == _GoogleDriveFile.TYPE_FOLDER and cname.isdigit():
            classes.append((cname.zfill(4), cid))
    classes.sort()
    print(f"discovered {len(classes)} numeric train classes", flush=True)
    if len(classes) < NUM_CLASSES:
        raise RuntimeError(f"Expected >= {NUM_CLASSES} train classes, found {len(classes)}")
    return classes[:NUM_CLASSES]


def ahash(rgb, side=12):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
    bits = small > small.mean()
    return bits.reshape(-1)


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def largest_face_crop(rgb):
    h0, w0 = rgb.shape[:2]
    # Speed up detection while preserving small faces.
    scale = min(1.0, 1100.0 / max(h0, w0))
    if scale < 1:
        det = cv2.resize(rgb, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        det = rgb
    gray = cv2.cvtColor(det, cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=4, minSize=(22, 22))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    if scale < 1:
        x, y, w, h = [int(round(v / scale)) for v in (x, y, w, h)]
    H, W = rgb.shape[:2]
    side = int(max(w, h) * 2.0)
    cx, cy = x + w // 2, y + h // 2
    x1, y1 = cx - side // 2, cy - side // 2
    x2, y2 = x1 + side, y1 + side
    # Pad with reflection if crop hits an image boundary.
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - W), max(0, y2 - H)
    if any((pad_l, pad_t, pad_r, pad_b)):
        rgb = cv2.copyMakeBorder(rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
        x1 += pad_l; x2 += pad_l; y1 += pad_t; y2 += pad_t
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return crop


def download_and_crop():
    classes = discover_train_classes()
    s = session()
    manifest = []
    counts = {}
    failure_counts = defaultdict(int)

    for yi, (cls, folder_id) in enumerate(classes):
        d = CROPS / cls
        d.mkdir(parents=True, exist_ok=True)
        _, children = list_folder(s, folder_id)
        files = []
        for fid, name, ftype, *_ in children:
            if ftype == _GoogleDriveFile.TYPE_FOLDER:
                continue
            if Path(name).suffix.lower() in IMG_EXT:
                files.append((fid, name))

        # Deterministic shuffle gives diversity while keeping runs reproducible.
        rng = random.Random(SEED + yi * 9973)
        rng.shuffle(files)
        candidates = files[: min(MAX_CANDIDATES_PER_CLASS, len(files))]
        seen_hashes = []
        accepted = 0

        for j, (fid, name) in enumerate(candidates):
            if accepted >= TARGET_PER_CLASS:
                break
            tmp = TMP / f"{cls}_{j:03d}{Path(name).suffix.lower() or '.jpg'}"
            try:
                result = gdown.download(
                    id=fid,
                    output=str(tmp),
                    quiet=True,
                    use_cookies=False,
                    timeout=45,
                    retries=3,
                )
                if not result or not tmp.exists():
                    failure_counts["download"] += 1
                    continue
                with Image.open(tmp) as im:
                    rgb = np.asarray(im.convert("RGB"))
                crop = largest_face_crop(rgb)
                if crop is None:
                    failure_counts["no_face"] += 1
                    continue
                ph = ahash(crop)
                if any(hamming(ph, old) <= 8 for old in seen_hashes):
                    failure_counts["near_duplicate"] += 1
                    continue
                seen_hashes.append(ph)
                outp = d / f"{accepted:03d}.jpg"
                Image.fromarray(crop).save(outp, quality=93)
                manifest.append({
                    "class_id": cls,
                    "drive_file_id": fid,
                    "source_filename": name,
                    "crop": "largest detected frontal face; 2.0x context square",
                })
                accepted += 1
            except Exception as e:
                failure_counts["exception"] += 1
                print(f"skip {cls} {name}: {type(e).__name__}: {e}", flush=True)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass

        counts[cls] = accepted
        print(f"class {cls}: accepted {accepted}/{TARGET_PER_CLASS} from {len(files)} available", flush=True)
        if accepted < MIN_PER_CLASS:
            raise RuntimeError(f"Class {cls} has only {accepted} usable faces")

    # No raw/source photos are uploaded. Keep only metadata sufficient to audit counts.
    safe_manifest = {
        "dataset": "KoIn50 normal-case train split",
        "official_root_id": KOIN50_ROOT_ID,
        "academic_use_notice": "KoIn official README states dataset/code are available for academic purposes.",
        "classes": len(classes),
        "accepted_images": len(manifest),
        "per_class_counts": counts,
        "failure_counts": dict(failure_counts),
        "raw_images_uploaded": False,
        "source_file_ids_uploaded": False,
    }
    (OUT / "data_summary.json").write_text(json.dumps(safe_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return classes, counts, manifest


class FaceDataset(Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(im).permute(2, 0, 1)
        if self.augment and torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[2])
        return x, torch.tensor(y, dtype=torch.long)


def make_split(classes):
    train, val = [], []
    for yi, (cls, _) in enumerate(classes):
        files = sorted((CROPS / cls).glob("*.jpg"))
        rng = random.Random(SEED + yi * 1231)
        rng.shuffle(files)
        nv = max(1, int(round(len(files) * 0.2)))
        val.extend((p, yi) for p in files[:nv])
        train.extend((p, yi) for p in files[nv:])
    random.Random(SEED).shuffle(train)
    random.Random(SEED + 1).shuffle(val)
    return train, val


class KoreanCelebCVAE(nn.Module):
    def __init__(self, nclass=NUM_CLASSES):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(16, 256), nn.SiLU(),
        )
        flat = 256 * 4 * 4
        self.mu = nn.Linear(flat, LATENT)
        self.lv = nn.Linear(flat, LATENT)
        self.id_head = nn.Sequential(nn.LayerNorm(LATENT), nn.Linear(LATENT, nclass))
        self.emb = nn.Embedding(nclass, EMBED)
        self.fc = nn.Linear(LATENT + EMBED, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )
    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.mu(h), self.lv(h)
    def decode(self, z, y):
        h = torch.cat([z, self.emb(y)], dim=1)
        h = self.fc(h).view(-1, 256, 4, 4)
        return self.dec(h)
    def forward(self, x, y):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z, y), mu, lv, self.id_head(mu)


def metrics(model, dl):
    model.eval()
    sums = defaultdict(float)
    n = 0
    with torch.no_grad():
        for x, y in dl:
            recon, mu, lv, logits = model(x, y)
            rec = F.l1_loss(recon, x, reduction="mean")
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            ce = F.cross_entropy(logits, y)
            pred = logits.argmax(1)
            b = x.size(0)
            sums["rec"] += float(rec) * b
            sums["kl"] += float(kl) * b
            sums["ce"] += float(ce) * b
            sums["acc"] += float((pred == y).float().sum())
            n += b
    return {"rec": sums["rec"] / n, "kl": sums["kl"] / n, "ce": sums["ce"] / n, "acc": sums["acc"] / n}


def save_grid(model, classes):
    model.eval()
    with torch.no_grad():
        y = torch.arange(NUM_CLASSES, dtype=torch.long)
        g = torch.Generator().manual_seed(SEED + 999)
        z = torch.randn(NUM_CLASSES, LATENT, generator=g)
        out = model.decode(z, y).clamp(-1, 1)
    cell = SIZE
    label_h = 16
    cols = 10
    rows = math.ceil(NUM_CLASSES / cols)
    canvas = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i in range(NUM_CLASSES):
        a = ((out[i].cpu() + 1) * 127.5).byte().permute(1, 2, 0).numpy()
        x = (i % cols) * cell
        y0 = (i // cols) * (cell + label_h)
        canvas.paste(Image.fromarray(a), (x, y0))
        draw.text((x + 2, y0 + cell + 2), classes[i][0], fill="black")
    canvas.save(OUT / "generated_koin50_samples.png")


def train(classes, counts):
    train_items, val_items = make_split(classes)
    train_ds = FaceDataset(train_items, augment=True)
    val_ds = FaceDataset(val_items, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = KoreanCelebCVAE()
    opt = torch.optim.AdamW(model.parameters(), lr=1.4e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1.5e-4)
    beta = 0.0005
    id_weight = 0.45
    log = []
    best = None
    best_score = -1e9
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = defaultdict(float); n = 0
        for x, y in train_dl:
            recon, mu, lv, logits = model(x, y)
            rec = F.l1_loss(recon, x)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            ce = F.cross_entropy(logits, y)
            loss = rec + beta * kl + id_weight * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            b = x.size(0); n += b
            total["loss"] += float(loss.detach()) * b
            total["rec"] += float(rec.detach()) * b
            total["kl"] += float(kl.detach()) * b
            total["ce"] += float(ce.detach()) * b
            total["acc"] += float((logits.argmax(1) == y).float().sum())
        sched.step()
        tr = {k: total[k] / n for k in ("loss", "rec", "kl", "ce", "acc")}
        va = metrics(model, val_dl)
        # Prioritize identity retention, then reconstruction quality.
        score = va["acc"] - 0.25 * va["rec"]
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train": tr, "val": va, "score": score}
        log.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best is not None:
        model.load_state_dict(best)
    elapsed = time.time() - t0
    final_val = metrics(model, val_dl)
    checkpoint = {
        "state_dict": model.state_dict(),
        "classes": [c for c, _ in classes],
        "class_semantics": "anonymous KoIn50 Korean celebrity identity IDs 0000-0049",
        "resolution": SIZE,
        "latent": LATENT,
        "embedding": EMBED,
        "seed": SEED,
        "training": {"epochs": EPOCHS, "elapsed_seconds": elapsed, "final_val": final_val},
        "academic_use_notice": "KoIn official README states dataset/code are publicly available for academic purposes.",
    }
    torch.save(checkpoint, OUT / "koin50_korean_celeb_cvae.pt")
    (OUT / "training_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    save_grid(model, classes)

    summary = {
        "dataset": "KoIn50 normal-case train split",
        "identity_classes": NUM_CLASSES,
        "accepted_face_crops": sum(counts.values()),
        "per_class_counts": counts,
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "resolution": SIZE,
        "latent": LATENT,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": EPOCHS,
        "elapsed_seconds_training_only": round(elapsed, 3),
        "validation_identity_accuracy": final_val["acc"],
        "chance_identity_accuracy": 1.0 / NUM_CLASSES,
        "validation_reconstruction_l1": final_val["rec"],
        "raw_training_photos_uploaded": False,
        "usage_scope": "KoIn README: academic purposes",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        classes, counts, manifest = download_and_crop()
        train(classes, counts)
    finally:
        # Ensure source images/crops are not retained in the uploaded artifact.
        shutil.rmtree(TMP, ignore_errors=True)
        shutil.rmtree(CROPS, ignore_errors=True)
