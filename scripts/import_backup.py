"""Restore observations from a backup ZIP made by the app's Export dialog.

  python scripts/import_backup.py ecoid-backup-20261007-101500.zip --app-data ~/.ecoid
Existing observations (same id) are left untouched; every photo is checked against its SHA-256; a bad record is
reported and skipped without stopping the rest. Close the app first.
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from app.observation.export import import_backup
from app.storage.store import ObservationStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backup")
    ap.add_argument("--app-data", default=str(Path.home() / ".ecoid"))
    a = ap.parse_args()
    res = import_backup(ObservationStore(a.app_data), a.backup)
    print(json.dumps(res["stats"]))
    for e in res["errors"][:20]:
        print("  problem:", e)


if __name__ == "__main__":
    main()
