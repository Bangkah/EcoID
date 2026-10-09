"""Turn the user's own VERIFIED / corrected field observations into dataset files (SRS: 'koleksi sendiri')."""
from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from app.ai.contract.labels import CLASSES
from app.ai.data.checks import read_metadata, write_metadata
from app.observation.models import Observation
from app.observation.verification import VerificationStatus
from app.storage.store import ObservationStore

NEGATIVE_DIR = {"evaluation": "negative", "val": "negative_val"}   # there is no negative TRAIN split (SRS section 6)


@dataclass(frozen=True)
class Item:
    obs: Observation
    dest_dir: str          # relative to data/, e.g. "evaluation/Mangifera_indica" or "negative/lookalike_plants"
    label: str


def plan_item(obs: Observation, split: str) -> tuple[Optional[Item], Optional[str]]:
    """-> (Item, None) or (None, reason it is skipped)."""
    if obs.verification_status == VerificationStatus.UNCERTAIN:
        return None, "uncertain: no trustworthy label"
    label = obs.human_label
    if obs.verification_status == VerificationStatus.REJECTED and not label:
        return None, "rejected without a correction: label unknown"
    if label in CLASSES:
        return Item(obs, f"{split}/{label.replace(' ', '_')}", label), None
    if split not in NEGATIVE_DIR:
        return None, "not one of the 5 classes and the train split has no negatives"
    return Item(obs, f"{NEGATIVE_DIR[split]}/lookalike_plants", ""), None


def export(store: ObservationStore, data_dir: Path, split: str, *, statuses=("VERIFIED", "REJECTED"),
           per_class: Optional[int] = None, strip_exif: bool = False, dry_run: bool = False):
    """Copy photos into data/<split>/…; idempotent (file name = observation id). Returns (Counter, [skipped reasons])."""
    if split not in ("train", "val", "evaluation"):
        raise ValueError("split must be train, val or evaluation")
    data_dir = Path(data_dir)
    meta = read_metadata(data_dir / "metadata" / "images.csv")
    stats, skipped, per_dest = Counter(), [], Counter()
    wanted = {VerificationStatus(s) for s in statuses}
    observations, _ = store.list(limit=200)
    page = 0
    while True:
        for obs in observations:
            if obs.verification_status not in wanted:
                stats["skipped_status_not_selected"] += 1
                continue
            item, reason = plan_item(obs, split)
            if item is None:
                stats["skipped"] += 1
                skipped.append(f"{obs.id}: {reason}")
                continue
            if per_class is not None and per_dest[item.dest_dir] >= per_class:
                stats["skipped_over_per_class_cap"] += 1
                continue
            src = store.abspath(obs.image_path)
            ext = ".jpg" if strip_exif else src.suffix
            dest = data_dir / item.dest_dir / f"{obs.id}{ext}"
            rel = dest.relative_to(data_dir).as_posix()
            per_dest[item.dest_dir] += 1
            if dest.exists():
                stats["already_exported"] += 1
                continue
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if strip_exif:
                    with Image.open(src) as im:
                        clean = ImageOps.exif_transpose(im).convert("RGB")     # keep orientation, drop all metadata
                    buf = io.BytesIO()
                    clean.save(buf, "JPEG", quality=95)
                    dest.write_bytes(buf.getvalue())
                else:
                    dest.write_bytes(src.read_bytes())
                meta[rel] = {"path": rel, "split": item.dest_dir.split("/")[0], "label": item.label,
                             "source": "EcoID field photo", "license": "own", "author": "(self)", "url": ""}
            stats["exported"] += 1
        page += 1
        if len(observations) < 200:
            break
        observations, _ = store.list(limit=200, offset=page * 200)
    if not dry_run and stats["exported"]:
        write_metadata(data_dir / "metadata" / "images.csv", meta.values())
    return stats, skipped
