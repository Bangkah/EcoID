"""Dataset quality checks (SRS section 6 rules), pure stdlib + Pillow.

Layout expected under data/:
  train/<Class_name>/*.jpg        val/<Class_name>/*.jpg
  evaluation/<Class_name>/*.jpg   (SRS: 10 per class, never seen in training)
  negative/<group>/*.jpg          negative_val/<group>/*.jpg
  metadata/images.csv             path,split,label,source,license,author,url
Class folder name = scientific name with '_' instead of ' '.
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.ai.contract.labels import CLASSES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_DIRS = tuple(c.replace(" ", "_") for c in CLASSES)
REQUIRED_NEGATIVE_GROUPS = ("lookalike_plants", "non_plant", "low_quality")


def list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def class_dir_to_label(name: str) -> str:
    return name.replace("_", " ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, size: int = 8) -> int:
    """64-bit difference hash; robust to resize/recompression."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.BILINEAR)
        px = np.asarray(im).reshape(-1).tolist()
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def find_corrupt(paths: list[Path]) -> list[Path]:
    bad = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.load()
        except Exception:
            bad.append(p)
    return bad


def find_duplicates(groups: dict[str, list[Path]], max_hamming: int = 5):
    """groups: split name -> paths. Returns (exact, near) lists of (pathA, pathB).

    Near-duplicates are reported across ALL splits, which catches train/eval leakage.
    """
    items = [(s, p) for s, ps in groups.items() for p in ps]
    by_hash: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for s, p in items:
        by_hash[sha256_file(p)].append((s, p))
    exact = [(g[0][1], g[i][1], g[0][0], g[i][0]) for g in by_hash.values() if len(g) > 1 for i in range(1, len(g))]

    hashes = []
    for s, p in items:
        try:
            hashes.append((dhash(p), s, p))
        except Exception:
            continue
    exact_pairs = {(a, b) for a, b, _, _ in exact}
    near = []
    for i in range(len(hashes)):
        hi, si, pi = hashes[i]
        for j in range(i + 1, len(hashes)):
            hj, sj, pj = hashes[j]
            if (hi ^ hj).bit_count() <= max_hamming and (pi, pj) not in exact_pairs and (pj, pi) not in exact_pairs:
                near.append((pi, pj, si, sj))
    return exact, near


def read_metadata(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["path"].replace("\\", "/"): row for row in csv.DictReader(f)}


def check_dataset(
    data_dir: Path,
    *,
    train_range=(300, 500),
    eval_per_class=10,
    min_negative=100,
    max_hamming=5,
    require_metadata=True,
) -> Report:
    r = Report()
    splits: dict[str, list[Path]] = {}

    for split in ("train", "val", "evaluation"):
        root = data_dir / split
        if not root.is_dir():
            (r.errors if split != "val" else r.warnings).append(f"missing folder: {split}/")
            continue
        found = {p.name for p in root.iterdir() if p.is_dir()}
        if found != set(CLASS_DIRS):
            r.errors.append(f"{split}/: class folders {sorted(found)} != expected {sorted(CLASS_DIRS)}")
        counts = {d: len(list_images(root / d)) for d in CLASS_DIRS}
        r.info.append(f"{split}: {counts}")
        splits[split] = list_images(root)
        for d, n in counts.items():
            if split == "train" and not (train_range[0] <= n <= train_range[1]):
                r.warnings.append(f"train/{d}: {n} images, SRS expects {train_range[0]}-{train_range[1]}")
            if split == "evaluation" and n != eval_per_class:
                r.warnings.append(f"evaluation/{d}: {n} images, SRS expects {eval_per_class}")
            if split == "val" and n == 0:
                r.errors.append(f"val/{d}: empty")
            if split == "train" and n == 0:
                r.errors.append(f"train/{d}: empty")

    for split in ("negative", "negative_val"):
        root = data_dir / split
        imgs = list_images(root)
        splits[split] = imgs
        if not root.is_dir():
            (r.errors if split == "negative" else r.warnings).append(f"missing folder: {split}/")
            continue
        r.info.append(f"{split}: {len(imgs)} images")
        if split == "negative" and len(imgs) < min_negative:
            r.errors.append(f"negative/: {len(imgs)} images, SRS requires >= {min_negative}")
        for g in REQUIRED_NEGATIVE_GROUPS:
            if split == "negative" and not list_images(root / g):
                r.errors.append(f"negative/{g}/ is missing or empty (SRS: must include lookalike plants, non-plants, low-quality)")

    allp = [p for ps in splits.values() for p in ps]
    for p in find_corrupt(allp):
        r.errors.append(f"corrupt/unreadable image: {p.relative_to(data_dir)}")

    exact, near = find_duplicates(splits, max_hamming)
    for a, b, sa, sb in exact:
        sev = r.errors if sa != sb else r.warnings
        sev.append(f"exact duplicate ({sa} vs {sb}): {a.relative_to(data_dir)} == {b.relative_to(data_dir)}")
    for a, b, sa, sb in near:
        sev = r.errors if sa != sb else r.warnings
        sev.append(f"near-duplicate ({sa} vs {sb}): {a.relative_to(data_dir)} ~ {b.relative_to(data_dir)}")

    meta = read_metadata(data_dir / "metadata" / "images.csv")
    if require_metadata:
        if not meta:
            r.errors.append("metadata/images.csv missing or empty")
        for p in allp:
            rel = p.relative_to(data_dir).as_posix()
            row = meta.get(rel)
            if row is None:
                r.errors.append(f"no metadata row: {rel}")
            elif not (row.get("license") or "").strip():
                r.errors.append(f"missing license: {rel}")
            elif "NC" in row["license"].upper().replace("-", " ").split():
                r.warnings.append(f"non-commercial license ({row['license']}): {rel}")
    return r
