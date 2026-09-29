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

OUT = Path("artifacts_v3")
OUT.mkdir(exist_ok=True)

SEED = 20260929
SIZE = 64
LATENT = 96
EMBED = 48
TOP_IDENTITIES = 24
MAX_PER_ID = 40
MIN_FACES = 20
EPOCHS = 18
BATCH = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


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
    counts = Counter(targets.tolist())
    selected_old = [k for k, _ in counts.most_common(TOP_IDENTITIES)]
    selected_old = sorted(selected_old, key=lambda k: (-counts[k], names[k]))
    old_to_new = {old: new for new, old in enumerate(selected_old)}

    rng = np.random.default_rng(SEED)
    all_images = []
    all_labels = []
    selected_names = []
    selected_counts = {}

    for old in selected_old:
        idx = np.flatnonzero(targets == old)
        rng.shuffle(idx)
        idx = idx[:MAX_PER_ID]
        selected_names.append(str(names[old]))
        selected_counts[str(names[old])] = int(len(idx))
        for i in idx:
            img = images[i]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            all_images.append(img)
            all_labels.append(old_to_new[old])

    X = np.stack(all_images, axis=0)
    y = np.asarray(all_labels, dtype=np.int64)
    return X, y, selected_names, selected_counts


def stratified_split(X, y, n_classes):
    rng = np.random.default_rng(SEED + 1)
    tr_idx, va_idx = [], []
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_val = max(3, int(round(len(idx) * 0.2)))
        n_val = min(n_val, len(idx) - 5)
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
            gain = random.uniform(0.92, 1.08)
            bias = random.uniform(-0.03, 0.03)
            a = np.clip(a * gain + bias, 0.0, 1.0)
        x = torch.from_numpy(a * 2.0 - 1.0).permute(2, 0, 1).float()
        return x, torch.tensor(self.y[i], dtype=torch.long)


class CVAEv3(nn.Module):
    def __init__(self, n_id):
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
        self.id_head = nn.Sequential(
            nn.Linear(LATENT, 128), nn.SiLU(), nn.Dropout(0.05), nn.Linear(128, n_id)
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
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.decode(z, y), mu, lv, self.id_head(mu)


def eval_model(model, loader):
    model.eval()
    sums = dict(loss=0.0, rec=0.0, kl=0.0, ce=0.0, acc=0.0)
    n = 0
    with torch.no_grad():
        for x, y in loader:
            recon, mu, lv, logits = model(x, y)
            rec = F.l1_loss(recon, x)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            ce = F.cross_entropy(logits, y)
            loss = rec + 0.003 * kl + 0.08 * ce
            acc = (logits.argmax(1) == y).float().mean()
            for k, v in [("loss", loss), ("rec", rec), ("kl", kl), ("ce", ce), ("acc", acc)]:
                sums[k] += float(v)
            n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


def save_samples(model, names, path):
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
            draw.text((c * SIZE + 2, r * (SIZE + 18) + SIZE + 1), name[:10], fill="black")
    canvas.save(path)


def main():
    X, y, names, source_counts = load_lfw()
    Xtr, ytr, Xva, yva = stratified_split(X, y, len(names))
    print(json.dumps({
        "selected_identities": len(names),
        "total_selected_images": int(len(X)),
        "train_images": int(len(Xtr)),
        "val_images": int(len(Xva)),
        "names": names,
        "source_counts": source_counts,
    }, indent=2))

    train_loader = DataLoader(Faces(Xtr, ytr, augment=True), batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(Faces(Xva, yva, augment=False), batch_size=BATCH, shuffle=False, num_workers=0)

    model = CVAEv3(len(names))
    opt = torch.optim.AdamW(model.parameters(), lr=1.4e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1.5e-4)
    log = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    step = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_sums = dict(loss=0.0, rec=0.0, kl=0.0, ce=0.0, acc=0.0)
        nb = 0
        for x, yb in train_loader:
            recon, mu, lv, logits = model(x, yb)
            rec = F.l1_loss(recon, x)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            ce = F.cross_entropy(logits, yb)
            acc = (logits.argmax(1) == yb).float().mean()
            step += 1
            beta = 0.003 * min(1.0, step / 80.0)
            loss = rec + beta * kl + 0.08 * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in [("loss", loss), ("rec", rec), ("kl", kl), ("ce", ce), ("acc", acc)]:
                train_sums[k] += float(v.detach())
            nb += 1
        sched.step()
        train_metrics = {k: v / nb for k, v in train_sums.items()}
        val_metrics = eval_model(model, val_loader)
        row = {
            "epoch": epoch,
            "step": step,
            "lr": opt.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        log.append(row)
        print(json.dumps(row))
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0
    final_val = eval_model(model, val_loader)

    checkpoint = {
        "format": "lfw-cvae-v3",
        "state_dict": model.state_dict(),
        "names": names,
        "resolution": SIZE,
        "latent": LATENT,
        "embedding": EMBED,
        "seed": SEED,
        "source_counts": source_counts,
        "training": {
            "epochs": EPOCHS,
            "steps": step,
            "elapsed_seconds": elapsed,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "best_val_metrics": final_val,
        },
    }
    torch.save(checkpoint, OUT / "lfw_cvae_v3.pt")
    save_samples(model, names, OUT / "generated_samples_v3.png")
    (OUT / "training_log_v3.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    summary = {
        "dataset": "Labeled Faces in the Wild (LFW), loaded through scikit-learn",
        "raw_dataset_artifact_uploaded": False,
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
        "best_val_loss": best_loss,
        "best_val_recon_l1": final_val["rec"],
        "best_val_identity_acc": final_val["acc"],
    }
    (OUT / "summary_v3.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
