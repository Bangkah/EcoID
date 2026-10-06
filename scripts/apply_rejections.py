"""Move images listed in data/metadata/reject.txt to data/_rejected/ (reversible) and forget their metadata.
  python scripts/apply_rejections.py --dry-run     # see what would move
Rejected photo ids are remembered, so `fetch_inat.py select` will not pick them again."""
import argparse
from pathlib import Path

import _common  # noqa: F401
from app.ai.data.qc import apply_rejections, read_reject_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--file", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    f = Path(a.file or Path(a.data) / "metadata" / "reject.txt")
    moved = apply_rejections(Path(a.data), read_reject_ids(f), dry_run=a.dry_run)
    print(("would move" if a.dry_run else "moved"), len(moved), "files")
    for m in moved:
        print("  ", m)


if __name__ == "__main__":
    main()
