"""Transfer-learning fine-tune of MobileNetV3 (SRS section 5.2).

Stage 1: freeze backbone, train the new 5-class head.
Stage 2: unfreeze the last N backbone blocks, lr 1e-4 with cosine decay.
Model selection uses data/val ONLY (never data/evaluation).

  python scripts/train.py --data data --arch small --out models/checkpoints/run1
"""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

import _torch_common as C
from app.ai.contract.labels import CLASSES


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fingerprint(ds):
    h = hashlib.sha256()
    for p, y in sorted(ds.items, key=lambda t: str(t[0])):
        h.update(f"{p.name}:{p.stat().st_size}:{y}\n".encode())
    return h.hexdigest()


def set_train_mode(model, first_trainable_block):
    """Train mode, but keep frozen blocks (incl. their BatchNorm stats) in eval mode."""
    model.train()
    for i, blk in enumerate(model.features):
        if i < first_trainable_block:
            blk.eval()


def run_epoch(model, loader, loss_fn, device, opt=None, first_trainable_block=0):
    training = opt is not None
    if training:
        set_train_mode(model, first_trainable_block)
    else:
        model.eval()
    tot_loss = correct = n = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return tot_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--arch", choices=["small", "large"], default="small")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights-path", default=None, help="local ImageNet state_dict (offline machines)")
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=20)
    ap.add_argument("--unfreeze-blocks", type=int, default=5, help="last N of model.features")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    seed_everything(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(a.data)
    train_ds = C.FolderDataset(data / "train", C.TRAIN_TF)
    val_ds = C.FolderDataset(data / "val", C.EVAL_TF)
    if not len(train_ds) or not len(val_ds):
        raise SystemExit("data/train and data/val must both contain images (run scripts/check_dataset.py).")
    print("train:", train_ds.counts())
    print("val:  ", val_ds.counts())

    g = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers, generator=g, drop_last=True)
    vl = DataLoader(val_ds, a.batch_size, shuffle=False, num_workers=a.workers)

    try:
        model = C.build_model(a.arch, pretrained=True, weights_path=a.weights_path).to(device)
    except Exception as e:  # SRS: training from scratch is not allowed
        raise SystemExit(f"Could not load ImageNet pretrained weights ({e}). "
                         "Use --weights-path on an offline machine. Training from scratch is not permitted.")

    n_blocks = len(model.features)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], {"val_acc": -1.0, "val_loss": 1e9, "epoch": -1}

    def stage(name, epochs, first_trainable_block, lr, cosine):
        nonlocal best
        for i, blk in enumerate(model.features):
            for p in blk.parameters():
                p.requires_grad = i >= first_trainable_block
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1)) if cosine else None
        bad = 0
        for ep in range(epochs):
            t0 = time.time()
            trl, tra = run_epoch(model, tl, loss_fn, device, opt, first_trainable_block)
            vl_, va = run_epoch(model, vl, loss_fn, device)
            if sched:
                sched.step()
            row = {"stage": name, "epoch": len(history), "train_loss": trl, "train_acc": tra,
                   "val_loss": vl_, "val_acc": va, "sec": time.time() - t0}
            history.append(row)
            print(json.dumps(row))
            if va > best["val_acc"] or (va == best["val_acc"] and vl_ < best["val_loss"]):
                best = {"val_acc": va, "val_loss": vl_, "epoch": row["epoch"], "stage": name}
                torch.save(model.state_dict(), out / "best.pt")
                bad = 0
            else:
                bad += 1
                if cosine and bad >= a.patience:
                    print("early stop")
                    break

    stage("head", a.epochs_head, n_blocks, a.lr_head, cosine=False)          # backbone frozen
    stage("finetune", a.epochs_finetune, max(n_blocks - a.unfreeze_blocks, 0), a.lr_finetune, cosine=True)

    meta = {
        "arch": f"mobilenet_v3_{a.arch}", "classes": list(CLASSES), "input_size": 224,
        "args": vars(a), "best": best, "history": history,
        "train_counts": train_ds.counts(), "val_counts": val_ds.counts(),
        "train_fingerprint": fingerprint(train_ds),
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "python": platform.python_version(), "device": str(device)},
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nbest val acc {best['val_acc']:.3f} at epoch {best['epoch']}  ->  {out/'best.pt'}")


if __name__ == "__main__":
    main()
