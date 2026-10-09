"""Optional OFFLINE background image for the map. EcoID never downloads map tiles.

<data_dir>/basemap.json:
  {"image": "basemap.png", "bounds": [south, west, north, east], "attribution": "© … (optional)"}
The image must be a Web-Mercator picture (what map screenshots/exports normally are) of exactly those bounds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

MAX_BYTES = 20 * 1024 * 1024
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
MAX_LAT = 85.0511


class BasemapError(ValueError):
    pass


def load_basemap(data_dir: Path) -> Optional[dict]:
    """None if not configured. Raises BasemapError (with a human-readable reason) if configured but unusable."""
    data_dir = Path(data_dir)
    cfg = data_dir / "basemap.json"
    if not cfg.is_file():
        return None
    try:
        c = json.loads(cfg.read_text(encoding="utf-8"))
        s, w, n, e = (float(x) for x in c["bounds"])
        name = str(c["image"])
    except (OSError, ValueError, KeyError, TypeError) as ex:
        raise BasemapError(f"basemap.json is not valid ({ex})") from None
    if not (-MAX_LAT <= s < n <= MAX_LAT and -180 <= w < e <= 180):
        raise BasemapError("bounds must be [south, west, north, east] with south<north, west<east, |lat|<=85.05")
    img = (data_dir / name).resolve()
    if data_dir.resolve() not in img.parents:
        raise BasemapError("the image must be inside the data folder")
    if img.suffix.lower() not in MIME:
        raise BasemapError("image must be .png, .jpg or .webp")
    if not img.is_file():
        raise BasemapError(f"image file not found: {name}")
    if img.stat().st_size > MAX_BYTES:
        raise BasemapError("image is larger than 20 MB")
    return {"image_path": img, "bounds": [s, w, n, e], "attribution": str(c.get("attribution") or ""),
            "mime": MIME[img.suffix.lower()]}
