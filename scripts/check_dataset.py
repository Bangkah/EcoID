"""Validate dataset layout, duplicates/leakage, licenses.  Exit code 1 on errors."""
import argparse
import sys

import _common  # noqa: F401
from app.ai.data.checks import check_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--max-hamming", type=int, default=5, help="near-duplicate dHash distance")
    ap.add_argument("--no-metadata", action="store_true", help="skip license/metadata checks (dev only)")
    a = ap.parse_args()
    from pathlib import Path
    r = check_dataset(Path(a.data), max_hamming=a.max_hamming, require_metadata=not a.no_metadata)
    for line in r.info:
        print("INFO   ", line)
    for line in r.warnings:
        print("WARNING", line)
    for line in r.errors:
        print("ERROR  ", line)
    print(f"\n{len(r.errors)} error(s), {len(r.warnings)} warning(s)")
    sys.exit(0 if r.ok else 1)


if __name__ == "__main__":
    main()
