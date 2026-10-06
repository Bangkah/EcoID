"""Contact sheets for fast manual label QC:  python scripts/make_contact_sheets.py --data data --split train
Open reports/contact_sheets/*.jpg, write the ids (captions) of wrong/ambiguous images into
data/metadata/reject.txt (one per line, '#' comments allowed), then run scripts/apply_rejections.py."""
import argparse
from pathlib import Path

import _common  # noqa: F401
from app.ai.data.checks import CLASS_DIRS
from app.ai.data.qc import make_sheets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", action="append", default=None, help="repeatable; default: train val evaluation")
    ap.add_argument("--out", default="reports/contact_sheets")
    ap.add_argument("--per-sheet", type=int, default=30)
    a = ap.parse_args()
    total = 0
    for split in a.split or ["train", "val", "evaluation"]:
        for d in CLASS_DIRS:
            folder = Path(a.data) / split / d
            if folder.is_dir():
                total += len(make_sheets(folder, Path(a.out), a.per_sheet))
    print(f"{total} sheets written to {a.out}")


if __name__ == "__main__":
    main()
