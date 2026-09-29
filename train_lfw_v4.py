import json
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_lfw_people

OUT = Path("artifacts_v4")
OUT.mkdir(exist_ok=True)

SEED = 20260929
SIZE = 64
LATENT = 96
EMBED = 48
TOP_IDENTITIES = 24
MAX_PER_ID = 40
MIN_FACES = 20
EPOCHS = 30
BATCH = 64
KL_MAX = 0.0005
ID_WEIGHT = 0.50

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def to_uint8(img):
    """scikit-learn LFW images are float32 scaled to [0,1]."""
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        vmax = float(np.nanmax(arr)) if arr.size else 0.0
        vmin = float(np.nanmin(arr)) if arr.size else 0.0
        if vmin >= -0.01 and vmax <= 1.5:
            arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def load_lfw():
    print("Downloading/loading LFW via scikit-learn...")
    data = fetch_lfw_people(
        funneled=True,
        resize=0.5,
        min_faces_per_person=MIN_FACES,
        color=True,
        download_if_missing=True,
        n_retries=5,
        delay=2.0,
    )
    images = data.images
    targets = data.target.astype(np.int64)
    names = np.asarray(data.target_names)

    print(json.dumps({
        "raw_dtype": str(images.dtype),
        "raw_min": float(images.min()),
        "raw_max": float(images.max()),
        "raw_shape": list(images.shape),
    }))

    counts = Counter(targets.tolist())
    selected_old = [k for k, _ in counts.most_common(TOP_IDENTITIES)]
    selected_old = sorted(selected_old, key=lambda k: (-counts[k], str(names[k])))
    old_to_new = {old: new for new, old in enumerate(selected_old)}

    rng = np.random.default_rng(SEED)
    all_images, all_labels = [], []
    selected_names = []
    selected_counts = {}

    for old in selected_old:
        idx = np.flatnonzero(targets == old)
        rng.shuffle(idx)
        idx = idx[:MAX_PER_ID]
        name = str(names[old])
        selected_names.append(name)
        selected_counts[name] = int(len(idx))
        for i in idx:
            img = to_uint8(images[i])
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
            all_images.append(img)
            all_labels.append(old_to_new[old])

    X = np.stack(all_images).astype(np.uint8)
    y = np.asarray(all_labels, dtype=np.int64)
    print(json.dumps({
        "post_dtype": str(X.dtype),
        "post_min": int(X.min()),
        "post_max": int(X.max()),
        "post_mean": float(X.mean()),
        "post_std": float(X.std()),
    }))
    return X, y, selected_names, selected_counts


def stratified_split(X, y, n_classes):
    rng = np.random.default_rng(SEED + 1)
    tr_idx, va_idx = [], []
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_val = max(4, int(round(len(idx) * 0.2)))
        n_val = min(n_val, len(idx) - 8)
        va_idx.extend(idx[:n_val].tolist())
        tr_idx.extend(idx[n_val:].tolist())
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


class Faces(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        a = self.X[i].astype(np.float32) / 255.0
        if self.augment:
            if random.random() < 0.5:
                a = a[:, ::-1].copy()
            gain = random.uniform(0.90, 1.10)
            bias = random.uniform(-0.035, 0.035)
            a = np.clip(a * gain + bias, 0.0, 1.0)
        x = torch.from_numpy(a * 2.0 - 1.0).permute(2, 0, 1).float()
        return x, torch.tensor(self.y[i], dtype=torch.long)


class CVAEv4(nn.Module):
    def __init__(self, n_id):
        super().__init__()
        self.n_id = n_id
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(16, 256), nn.SiLU(),
        )
        flat = 256 * 4 * 4
        self.mu = nn.Linear(flat, LATENT)
        self.lv = nn.Linear(flat, LATENT)
        self.id_head = nn.Sequential(
            nn.Linear(LATENT, 192), nn.SiLU(), nn.Dropout(0.10), nn.Linear(192, n_id)
        )
        self.emb = nn.Embedding(n_id, EMBED)
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
        return self.dec(self.fc(h).view(-1, 256, 4, 4))

    def forward(self, x, y):
        mu, lv = self.encode(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z, y)
        logits = self.id_head(mu)
        return recon, mu, lv, logits


def loss_parts(model, x, y, step=None):
    recon, mu, lv, logits = model(x, y)
    rec = F.l1_loss(recon, x)
    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    ce = F.cross_entropy(logits, y)
    acc = (logits.argmax(1) == y).float().mean()
    if step is None:
        beta = KL_MAX
    else:
        beta = KL_MAX * min(1.0, step / 120.0)
    loss = rec + beta * kl + ID_WEIGHT * ce
    return loss, rec, kl, ce, acc


def evaluate(model, loader):
    model.eval()
    sums = {"loss": 0.0, "rec": 0.0, "kl": 0.0, "ce": 0.0, "acc": 0.0}
    n_batches = 0
    n_samples = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            recon, mu, lv, logits = model(x, y)
            rec = F.l1_loss(recon, x)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            ce = F.cross_entropy(logits, y)
            loss = rec + KL_MAX * kl + ID_WEIGHT * ce
            pred = logits.argmax(1)
            correct += int((pred == y).sum())
            n_samples += int(y.numel())
            sums["loss"] += float(loss)
            sums["rec"] += float(rec)
            sums["kl"] += float(kl)
            sums["ce"] += float(ce)
            n_batches += 1
    return {
        "loss": sums["loss"] / max(1, n_batches),
        "rec": sums["rec"] / max(1, n_batches),
        "kl": sums["kl"] / max(1, n_batches),
        "ce": sums["ce"] / max(1, n_batches),
        "acc": correct / max(1, n_samples),
    }


def save_recon_grid(model, X, y, names, path, n=18):
    model.eval()
    n = min(n, len(X))
    rng = np.random.default_rng(SEED + 44)
    idx = rng.choice(len(X), size=n, replace=False)
    cols = 6
    rows = int(np.ceil(n / cols))
    cell_h = SIZE * 2 + 18
    canvas = Image.new("RGB", (cols * SIZE, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    with torch.no_grad():
        for j, i in enumerate(idx):
            orig = X[i]
            x = torch.from_numpy(orig.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
            yy = torch.tensor([int(y[i])], dtype=torch.long)
            mu, _ = model.encode(x)
            rec = model.decode(mu, yy)[0]
            arr = ((rec.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            r, c = divmod(j, cols)
            y0 = r * cell_h
            canvas.paste(Image.fromarray(orig), (c * SIZE, y0))
            canvas.paste(Image.fromarray(arr), (c * SIZE, y0 + SIZE))
            draw.text((c * SIZE + 1, y0 + SIZE * 2), names[int(y[i])][:9], fill="black")
    canvas.save(path)


def save_random_grid(model, names, path):
    model.eval()
    cols = 6
    rows = int(np.ceil(len(names) / cols))
    canvas = Image.new("RGB", (cols * SIZE, rows * (SIZE + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    g = torch.Generator().manual_seed(SEED + 777)
    with torch.no_grad():
        for i, name in enumerate(names):
            y = torch.tensor([i], dtype=torch.long)
            z = torch.randn(1, LATENT, generator=g)
            t = model.decode(z, y)[0]
            arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            r, c = divmod(i, cols)
            canvas.paste(Image.fromarray(arr), (c * SIZE, r * (SIZE + 18)))
            draw.text((c * SIZE + 1, r * (SIZE + 18) + SIZE), name[:9], fill="black")
    canvas.save(path)


def main():
    X, y, names, source_counts = load_lfw()
    Xtr, ytr, Xva, yva = stratified_split(X, y, len(names))
    print(json.dumps({
        "identities": len(names),
        "selected_images": int(len(X)),
        "train_images": int(len(Xtr)),
        "val_images": int(len(Xva)),
        "chance_acc": 1.0 / len(names),
        "names": names,
    }, indent=2))

    train_loader = DataLoader(Faces(Xtr, ytr, augment=True), batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(Faces(Xva, yva, augment=False), batch_size=BATCH, shuffle=False, num_workers=0)

    model = CVAEv4(len(names))
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1.5e-4)
    log = []
    best_loss = float("inf")
    best_acc = -1.0
    best_state = None
    best_epoch = 0
    step = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        sums = {"loss": 0.0, "rec": 0.0, "kl": 0.0, "ce": 0.0, "acc": 0.0}
        nb = 0
        train_correct = 0
        train_n = 0
        for x, yb in train_loader:
            recon, mu, lv, logits = model(x, yb)
            rec = F.l1_loss(recon, x)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            ce = F.cross_entropy(logits, yb)
            step += 1
            beta = KL_MAX * min(1.0, step / 120.0)
            loss = rec + beta * kl + ID_WEIGHT * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pred = logits.argmax(1)
            train_correct += int((pred == yb).sum())
            train_n += int(yb.numel())
            sums["loss"] += float(loss.detach())
            sums["rec"] += float(rec.detach())
            sums["kl"] += float(kl.detach())
            sums["ce"] += float(ce.detach())
            nb += 1
        sched.step()
        train_metrics = {
            "loss": sums["loss"] / nb,
            "rec": sums["rec"] / nb,
            "kl": sums["kl"] / nb,
            "ce": sums["ce"] / nb,
            "acc": train_correct / max(1, train_n),
        }
        val_metrics = evaluate(model, val_loader)
        row = {"epoch": epoch, "step": step, "lr": opt.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        log.append(row)
        print(json.dumps(row))

        # Prefer identity accuracy; break ties with total validation objective.
        if (val_metrics["acc"] > best_acc + 1e-12) or (
            abs(val_metrics["acc"] - best_acc) <= 1e-12 and val_metrics["loss"] < best_loss
        ):
            best_acc = val_metrics["acc"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0
    best_val = evaluate(model, val_loader)

    torch.save({
        "format": "lfw-cvae-v4-fixed-scale",
        "state_dict": model.state_dict(),
        "names": names,
        "resolution": SIZE,
        "latent": LATENT,
        "embedding": EMBED,
        "seed": SEED,
        "source_counts": source_counts,
        "preprocessing": "scikit-learn [0,1] float LFW -> multiply 255 -> uint8 -> resize 64 -> normalize to [-1,1]",
        "training": {
            "epochs": EPOCHS,
            "steps": step,
            "elapsed_seconds": elapsed,
            "best_epoch": best_epoch,
            "best_val": best_val,
            "kl_max": KL_MAX,
            "identity_weight": ID_WEIGHT,
        },
    }, OUT / "lfw_cvae_v4.pt")

    save_recon_grid(model, Xva, yva, names, OUT / "reconstruction_grid_v4.png")
    save_random_grid(model, names, OUT / "generated_samples_v4.png")
    (OUT / "training_log_v4.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    summary = {
        "dataset": "Labeled Faces in the Wild (LFW), scikit-learn loader",
        "identities": len(names),
        "identity_names": names,
        "source_images_selected": int(len(X)),
        "train_images": int(len(Xtr)),
        "validation_images": int(len(Xva)),
        "resolution": SIZE,
        "latent": LATENT,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": EPOCHS,
        "steps": step,
        "elapsed_seconds": round(elapsed, 3),
        "best_epoch": best_epoch,
        "best_val_loss": best_val["loss"],
        "best_val_recon_l1": best_val["rec"],
        "best_val_identity_acc": best_val["acc"],
        "chance_identity_acc": 1.0 / len(names),
        "pixel_scale_check": {"uint8_min": int(X.min()), "uint8_max": int(X.max()), "mean": float(X.mean()), "std": float(X.std())},
        "raw_dataset_artifact_uploaded": False,
    }
    (OUT / "summary_v4.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
