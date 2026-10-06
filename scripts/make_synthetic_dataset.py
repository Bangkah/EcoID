"""Generate a tiny SYNTHETIC dataset with the exact layout check_dataset.py expects.

Purpose: rehearse / smoke-test the whole Phase 2 pipeline (check -> train -> export -> calibrate -> evaluate)
without real photos. The images are coloured noise: they prove the plumbing works, NOT that plants are recognised.

  python scripts/make_synthetic_dataset.py --out data_synth
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import _common  # noqa: F401
from app.ai.data.checks import CLASS_DIRS, list_images

COLORS = [(210, 60, 50), (60, 190, 70), (60, 90, 210), (220, 200, 60), (170, 70, 190)]  # one hue per class
NEG_GROUPS = ("lookalike_plants", "non_plant", "low_quality")


def save(rng, path: Path, base, size=96, noise=45):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(np.array(base) + rng.normal(0, noise, (size, size, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path, "JPEG", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--train-per-class", type=int, default=24)
    ap.add_argument("--val-per-class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)

    for split, n in (("train", a.train_per_class), ("val", a.val_per_class), ("evaluation", 10)):
        for ci, d in enumerate(CLASS_DIRS):
            for i in range(n):
                save(rng, out / split / d / f"{i:03d}.jpg", COLORS[ci])
    for split, n in (("negative", 40), ("negative_val", 12)):          # 3 groups x 40 = 120 >= 100 (SRS)
        for g in NEG_GROUPS:
            for i in range(n):
                base = [tuple(rng.integers(90, 170, 3))] if g != "low_quality" else [(128, 128, 128)]
                save(rng, out / split / g / f"{i:03d}.jpg", base[0], noise=60 if g != "low_quality" else 10)

    (out / "metadata").mkdir(parents=True, exist_ok=True)
    with open(out / "metadata" / "images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["path", "split", "label", "source", "license", "author", "url"])
        w.writeheader()
        for p in list_images(out):
            rel = p.relative_to(out).as_posix()
            w.writerow({"path": rel, "split": rel.split("/")[0], "label": "", "source": "synthetic",
                        "license": "CC0", "author": "make_synthetic_dataset.py", "url": ""})
    print(f"synthetic dataset written to {out} ({len(list_images(out))} images)")


if __name__ == "__main__":
    main()
