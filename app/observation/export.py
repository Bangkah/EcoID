"""Export (CSV / GeoJSON / JSON / full backup ZIP) and restore. Standard library + Pillow only; no network."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from app.ai.contract.labels import COMMON_NAMES
from app.observation.manager import ObservationManager
from app.observation.models import Observation
from app.storage.store import NotFoundError, ObservationStore

CSV_FIELDS = ["id", "timestamp", "captured_at", "predicted_species", "common_name", "confidence",
              "identification_status", "verification_status", "human_label", "user_species", "latitude", "longitude",
              "location_source", "location_accuracy_m", "notes", "model", "photo"]
_LOC_KEYS = ("latitude", "longitude", "location_source", "location_accuracy_m")
BACKUP_FORMAT = "ecoid-backup"
BACKUP_VERSION = 1
_ID = re.compile(r"^[0-9a-f]{32}$")
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024


def _row(o: Observation, include_location: bool) -> dict:
    d = o.to_dict()
    if not include_location:
        for k in _LOC_KEYS:
            d[k] = None
    return d


def _csv_safe(v):
    """Spreadsheets run text starting with = + - @ as a formula. Neutralise it with a leading apostrophe."""
    if isinstance(v, str) and v and v[0] in "=+-@\t\r":
        return "'" + v
    return v


def to_csv(observations, include_location=True) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, CSV_FIELDS)
    w.writeheader()
    for o in observations:
        d = _row(o, include_location)
        d["common_name"] = COMMON_NAMES.get(o.predicted_species, "")
        d["photo"] = Path(o.image_path).name
        w.writerow({k: _csv_safe("" if d.get(k) is None else d.get(k)) for k in CSV_FIELDS})
    return ("\ufeff" + buf.getvalue()).encode("utf-8")           # BOM so Excel reads Indonesian text correctly


def to_geojson(observations, include_location=True) -> bytes:
    """RFC 7946: coordinates are [longitude, latitude]. Needs locations, so include_location=False is an error."""
    if not include_location:
        raise ValueError("GeoJSON is all about locations: it cannot be exported without them")
    feats = []
    for o in observations:
        if o.latitude is None:
            continue
        props = {k: v for k, v in o.to_dict().items() if k not in ("latitude", "longitude", "image_path")}
        props["common_name"] = COMMON_NAMES.get(o.predicted_species)
        props["photo"] = Path(o.image_path).name
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [o.longitude, o.latitude]},
                      "properties": props})
    return json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False, indent=1).encode("utf-8")


def to_json(observations, include_location=True) -> bytes:
    return json.dumps([_row(o, include_location) for o in observations], ensure_ascii=False, indent=1).encode("utf-8")


# ----------------------------------------------------------------------------- backup / restore
def build_backup(store: ObservationStore, observations, fileobj, include_location=True) -> dict:
    """Write a ZIP: manifest.json, observations.json, images/<id><ext> (originals, untouched). Returns the manifest."""
    files, rows = [], []
    with zipfile.ZipFile(fileobj, "w", zipfile.ZIP_STORED) as z:       # photos are already compressed
        for o in observations:
            src = store.abspath(o.image_path)
            if not src.is_file():
                continue
            data = src.read_bytes()
            name = f"images/{o.id}{src.suffix.lower()}"
            z.writestr(name, data)
            files.append({"id": o.id, "name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            rows.append(_row(o, include_location))
        manifest = {"format": BACKUP_FORMAT, "version": BACKUP_VERSION, "created": datetime.now(timezone.utc).isoformat(),
                    "count": len(rows), "includes_location": include_location, "files": files}
        z.writestr("observations.json", json.dumps(rows, ensure_ascii=False))
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
    return manifest


def _read_member(z: zipfile.ZipFile, name: str, limit: int) -> bytes:
    info = z.getinfo(name)
    if info.file_size > limit:
        raise ValueError(f"{name} is too large")
    return z.read(name)


def import_backup(store: ObservationStore, path) -> dict:
    """Restore observations from a backup ZIP. Existing ids are left alone. Never trusts paths inside the archive:
    photos are matched by observation id and written to a location computed here (no zip-slip)."""
    stats, errors = Counter(), []
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if "manifest.json" not in names or "observations.json" not in names:
            raise ValueError("not an EcoID backup (manifest.json / observations.json missing)")
        manifest = json.loads(_read_member(z, "manifest.json", MAX_JSON_BYTES))
        if manifest.get("format") != BACKUP_FORMAT or manifest.get("version") != BACKUP_VERSION:
            raise ValueError("unsupported backup format/version")
        by_id = {f["id"]: f for f in manifest.get("files", []) if isinstance(f, dict) and _ID.match(str(f.get("id", "")))}
        for d in json.loads(_read_member(z, "observations.json", MAX_JSON_BYTES)):
            oid = d.get("id") if isinstance(d, dict) else None
            try:
                if not isinstance(oid, str) or not _ID.match(oid):
                    raise ValueError("bad observation id")
                obs = Observation.from_dict(d)
                try:
                    store.get(oid)
                    stats["skipped_existing"] += 1
                    continue
                except NotFoundError:
                    pass
                f = by_id.get(oid)
                if not f or f["name"] not in names:
                    raise ValueError("photo missing from the archive")
                ext = Path(f["name"]).suffix.lower()
                if ext not in _EXTS or f["name"] != f"images/{oid}{ext}":
                    raise ValueError("unexpected photo name")
                data = _read_member(z, f["name"], MAX_IMAGE_BYTES)
                if hashlib.sha256(data).hexdigest() != f["sha256"]:
                    raise ValueError("photo does not match its checksum")
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                try:
                    when = datetime.fromisoformat(obs.timestamp)
                except ValueError:
                    when = datetime.now(timezone.utc)
                rel = f"images/{when:%Y}/{when:%m}/{oid}{ext}"
                dest = store.abspath(rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                try:
                    ObservationManager._write_thumb(data, store.thumb_path(oid))
                    store.insert(replace(obs, image_path=rel))
                except Exception:
                    dest.unlink(missing_ok=True)
                    store.thumb_path(oid).unlink(missing_ok=True)
                    raise
                stats["imported"] += 1
            except Exception as e:                    # one bad record must not abort the rest
                stats["rejected"] += 1
                errors.append(f"{oid}: {e}")
    return {"stats": dict(stats), "errors": errors}
