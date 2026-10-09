"""Export your own field observations into the dataset (closes the loop: field photos -> evaluation/training data).

  python scripts/export_observations.py --split evaluation --per-class 10
  python scripts/export_observations.py --split val --dry-run

What is exported: VERIFIED (label = the suggestion you confirmed) and REJECTED *with* a correction
(label = what you said it was; if that is not one of the 5 classes it becomes a lookalike NEGATIVE).
Never exported: UNCERTAIN, and REJECTED without a correction.
Choose ONE split per place/garden: photos of the same plants in train and evaluation would leak.
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
from app.ai.data.field_export import export
from app.storage.store import ObservationStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-data", default=str(Path.home() / ".ecoid"), help="the EcoID app's --data-dir")
    ap.add_argument("--data", default="data", help="the dataset folder")
    ap.add_argument("--split", required=True, choices=["train", "val", "evaluation"])
    ap.add_argument("--status", default="VERIFIED,REJECTED")
    ap.add_argument("--per-class", type=int, default=None, help="cap per destination folder (SRS evaluation: 10)")
    ap.add_argument("--strip-exif", action="store_true", help="re-encode as JPEG without metadata (removes GPS)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (Path(a.app_data) / "ecoid.db").is_file():
        raise SystemExit(f"no EcoID database in {a.app_data}")
    stats, skipped = export(ObservationStore(a.app_data), Path(a.data), a.split, statuses=tuple(a.status.split(",")),
                            per_class=a.per_class, strip_exif=a.strip_exif, dry_run=a.dry_run)
    print(("DRY RUN " if a.dry_run else "") + str(dict(stats)))
    for s in skipped[:20]:
        print("  skipped", s)
    if not a.dry_run:
        print("Next: python scripts/check_dataset.py   (checks duplicates/leakage against the rest of the dataset)")


if __name__ == "__main__":
    main()
