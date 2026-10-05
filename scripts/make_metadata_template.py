"""Scan data/ and (re)create metadata/images.csv, keeping rows you already filled in.
Fill in source/license/author/url by hand (or from your download script), then run check_dataset.py."""
import csv
from pathlib import Path

import _common  # noqa: F401
from app.ai.data.checks import list_images, read_metadata

FIELDS = ["path", "split", "label", "source", "license", "author", "url"]


def main(data=Path("data")):
    existing = read_metadata(data / "metadata" / "images.csv")
    rows = []
    for p in list_images(data):
        rel = p.relative_to(data).as_posix()
        parts = rel.split("/")
        if parts[0] == "metadata":
            continue
        row = existing.get(rel) or {"path": rel, "split": parts[0],
                                    "label": parts[1].replace("_", " ") if parts[0] in ("train", "val", "evaluation") else "",
                                    "source": "", "license": "", "author": "", "url": ""}
        rows.append({k: row.get(k, "") for k in FIELDS})
    (data / "metadata").mkdir(parents=True, exist_ok=True)
    with open(data / "metadata" / "images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    blank = sum(1 for r in rows if not r["license"])
    print(f"{len(rows)} rows written, {blank} still need a license")


if __name__ == "__main__":
    main()
