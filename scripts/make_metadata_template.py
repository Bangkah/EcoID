"""Scan data/ and (re)create metadata/images.csv, keeping rows you already filled in.
Fill in source/license/author/url by hand, then run check_dataset.py.

  --own PREFIX   mark every row under PREFIX that has no licence as your OWN photo (source=own, license=own),
                 e.g. --own negative/non_plant   (repeatable)
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
from app.ai.data.checks import list_images, read_metadata, write_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--own", action="append", default=[], metavar="PREFIX")
    a = ap.parse_args()
    data = Path(a.data)
    existing = read_metadata(data / "metadata" / "images.csv")
    rows = []
    for p in list_images(data):
        rel = p.relative_to(data).as_posix()
        parts = rel.split("/")
        if parts[0] == "metadata" or parts[0].startswith("_"):
            continue
        row = dict(existing.get(rel) or {"path": rel, "split": parts[0],
                                         "label": parts[1].replace("_", " ") if parts[0] in ("train", "val", "evaluation") else "",
                                         "source": "", "license": "", "author": "", "url": ""})
        if not (row.get("license") or "").strip() and any(rel.startswith(x.rstrip("/") + "/") for x in a.own):
            row.update(source="own", license="own", author="(self)")
        rows.append(row)
    write_metadata(data / "metadata" / "images.csv", rows)
    blank = sum(1 for r in rows if not (r.get("license") or "").strip())
    print(f"{len(rows)} rows written, {blank} still need a license")


if __name__ == "__main__":
    main()
