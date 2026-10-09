"""Optional metadata embedded in a photo (phone cameras write it): where and when it was taken."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional

from PIL import Image

_GPS_IFD, _EXIF_IFD = 0x8825, 0x8769
_DT_ORIGINAL, _OFFSET_ORIGINAL = 0x9003, 0x9011


@dataclass(frozen=True)
class ExifInfo:
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    captured_at: Optional[str] = None      # ISO-8601; has an offset only if the camera recorded one

    @property
    def has_location(self) -> bool:
        return self.latitude is not None

    def to_dict(self) -> dict:
        return {"latitude": self.latitude, "longitude": self.longitude, "captured_at": self.captured_at}


def _dms(v) -> Optional[float]:
    try:
        d, m, s = (float(x) for x in v)
    except (TypeError, ValueError):
        return None
    return d + m / 60.0 + s / 3600.0


def read_exif(data: bytes) -> ExifInfo:
    """Never raises: broken or missing metadata simply yields an empty ExifInfo."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            exif = im.getexif()
            gps = dict(exif.get_ifd(_GPS_IFD))
            sub = dict(exif.get_ifd(_EXIF_IFD))
    except Exception:
        return ExifInfo()

    lat = lon = None
    la, lo = _dms(gps.get(2)), _dms(gps.get(4))
    if la is not None and lo is not None:
        la = -la if str(gps.get(1, "N")).upper().startswith("S") else la
        lo = -lo if str(gps.get(3, "E")).upper().startswith("W") else lo
        # (0, 0) is what many phones write when they had no fix: not a real position
        if -90 <= la <= 90 and -180 <= lo <= 180 and not (la == 0 and lo == 0):
            lat, lon = round(la, 6), round(lo, 6)

    when = None
    raw = sub.get(_DT_ORIGINAL)
    if isinstance(raw, str):
        m = re.match(r"^\s*(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", raw)
        if m and m.group(1) != "0000":
            when = "{}-{}-{}T{}:{}:{}".format(*m.groups())
            off = sub.get(_OFFSET_ORIGINAL)
            if isinstance(off, str) and re.match(r"^[+-]\d{2}:\d{2}$", off.strip()):
                when += off.strip()
    return ExifInfo(lat, lon, when)
