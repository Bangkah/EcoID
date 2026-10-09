"""Observation Manager + Verification Manager (SRS section 15).

Flow (SRS section 16):  photo -> identify -> DRAFT -> human verification -> saved Observation.
A draft is a photo + model suggestion waiting for the user to look at the real plant.
Nothing becomes an observation until the user chooses VERIFIED / REJECTED / UNCERTAIN.
"""
from __future__ import annotations

import io
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from app.ai.contract.result import Candidate, IdentificationResult, Status
from app.ai.inference.identifier import Identifier
from app.ai.preprocessing.preprocess import InvalidImageError, load_image
from app.observation.exif import ExifInfo, read_exif
from app.observation.models import Observation
from app.observation.species import normalize_species
from app.observation.verification import parse_status
from app.storage.store import UNSET, NotFoundError, ObservationStore

MAX_IMAGE_BYTES = 25 * 1024 * 1024
_FORMAT_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
THUMB_SIZE = (320, 320)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _number(v, name):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(v)


def parse_location(latitude, longitude, source=None, accuracy=None):
    """-> None (no location) or (lat, lon, source, accuracy_m); raises ValueError on anything malformed."""
    lat, lon, acc = _number(latitude, "latitude"), _number(longitude, "longitude"), _number(accuracy, "location_accuracy_m")
    if lat is None and lon is None:
        return None
    if lat is None or lon is None:
        raise ValueError("latitude and longitude must be given together")
    return lat, lon, source or "manual", acc


@dataclass(frozen=True)
class Draft:
    id: str
    result: IdentificationResult
    ext: str
    created: float
    exif: ExifInfo = ExifInfo()

    def to_dict(self) -> dict:
        return {"draft_id": self.id, "result": self.result.to_dict(), "threshold": self.result.threshold,
                "exif": self.exif.to_dict()}


class ObservationManager:
    def __init__(self, identifier: Identifier, store: ObservationStore, *,
                 max_image_bytes: int = MAX_IMAGE_BYTES,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self._ident = identifier
        self.store = store
        self._max = max_image_bytes
        self._clock = clock
        self._infer_lock = threading.Lock()   # one inference at a time: predictable CPU use

    # ---- drafts ------------------------------------------------------------------------
    def _draft_files(self, draft_id: str):
        if not _ID_RE.match(draft_id or ""):
            raise NotFoundError(draft_id)
        d = self.store.data_dir / "drafts"
        return d / f"{draft_id}.json", d

    def identify(self, image_bytes: bytes) -> Draft:
        if not image_bytes:
            raise InvalidImageError("Empty upload")
        if len(image_bytes) > self._max:
            raise InvalidImageError(f"Image larger than {self._max // (1024 * 1024)} MB")
        try:
            with Image.open(io.BytesIO(image_bytes)) as im:
                fmt = im.format
        except Exception as e:
            raise InvalidImageError(f"Cannot decode image: {e}") from e
        ext = _FORMAT_EXT.get(fmt or "")
        if ext is None:
            raise InvalidImageError(f"Unsupported image format: {fmt}")
        try:
            with self._infer_lock:
                result = self._ident.identify(image_bytes)   # decodes fully; raises InvalidImageError
        except Image.DecompressionBombError as e:
            raise InvalidImageError("Image dimensions are too large") from e

        exif = read_exif(image_bytes)
        draft_id = uuid.uuid4().hex
        meta_path, ddir = self._draft_files(draft_id)
        (ddir / f"{draft_id}{ext}").write_bytes(image_bytes)           # original bytes, untouched
        meta_path.write_text(json.dumps({"result": result.to_dict(), "threshold": result.threshold,
                                         "ext": ext, "created": time.time(), "exif": exif.to_dict()}), encoding="utf-8")
        return Draft(draft_id, result, ext, time.time(), exif)

    def _load_draft(self, draft_id: str) -> Draft:
        meta_path, ddir = self._draft_files(draft_id)
        if not meta_path.is_file():
            raise NotFoundError(draft_id)
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        r = m["result"]
        result = IdentificationResult(
            candidates=tuple(Candidate(c["label"], c["score"]) for c in r["candidates"]),
            status=Status(r["status"]), model=r["model"], threshold=m["threshold"])
        e = m.get("exif") or {}                       # drafts written by older versions have no "exif"
        return Draft(draft_id, result, m["ext"], m["created"],
                     ExifInfo(e.get("latitude"), e.get("longitude"), e.get("captured_at")))

    def discard(self, draft_id: str) -> None:
        meta_path, ddir = self._draft_files(draft_id)
        draft = self._load_draft(draft_id)
        meta_path.unlink(missing_ok=True)
        (ddir / f"{draft_id}{draft.ext}").unlink(missing_ok=True)

    def purge_drafts(self, max_age_hours: float = 24.0) -> int:
        """Remove abandoned drafts (called at startup)."""
        n, cutoff = 0, time.time() - max_age_hours * 3600
        for meta in (self.store.data_dir / "drafts").glob("*.json"):
            try:
                if json.loads(meta.read_text(encoding="utf-8")).get("created", 0) < cutoff:
                    self.discard(meta.stem)
                    n += 1
            except Exception:
                meta.unlink(missing_ok=True)
        return n

    # ---- save (after human verification) --------------------------------------------------
    def save(self, draft_id: str, verification_status, notes: str = "",
             latitude: Optional[float] = None, longitude: Optional[float] = None, *,
             location_source: Optional[str] = None, location_accuracy_m: Optional[float] = None,
             user_species=None, use_photo_location: bool = False) -> Observation:
        """Location is opt-in: it is stored only if the caller passes coordinates, or asks for the photo's own EXIF
        position with use_photo_location=True. The capture time from EXIF is always kept (it is not a place)."""
        status = parse_status(verification_status)
        draft = self._load_draft(draft_id)
        _, ddir = self._draft_files(draft_id)
        src = ddir / f"{draft_id}{draft.ext}"
        if not src.is_file():
            raise NotFoundError(draft_id)

        now = self._clock()
        obs_id = uuid.uuid4().hex
        rel = f"images/{now:%Y}/{now:%m}/{obs_id}{draft.ext}"
        top, *alts = draft.result.candidates
        loc = parse_location(latitude, longitude, location_source, location_accuracy_m)
        if loc is None and use_photo_location:
            if not draft.exif.has_location:
                raise ValueError("this photo has no location information")
            loc = (draft.exif.latitude, draft.exif.longitude, "exif", None)
        lat, lon, loc_src, acc = loc or (None, None, None, None)
        obs = Observation(                                   # validates before any file is moved
            id=obs_id, image_path=rel, predicted_species=top.label, confidence=top.score,
            alternative_predictions=tuple({"label": c.label, "score": c.score} for c in alts),
            verification_status=status, timestamp=now.isoformat(), latitude=lat, longitude=lon,
            notes=(notes or "").strip(), model=draft.result.model, identification_status=draft.result.status,
            user_species=normalize_species(user_species), captured_at=draft.exif.captured_at,
            location_source=loc_src, location_accuracy_m=acc)

        dest = self.store.abspath(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        dest.write_bytes(data)                               # full-resolution original is kept
        try:
            self._write_thumb(data, self.store.thumb_path(obs_id))
            self.store.insert(obs)
        except Exception:
            dest.unlink(missing_ok=True)
            self.store.thumb_path(obs_id).unlink(missing_ok=True)
            raise
        self.discard(draft_id)
        return obs

    @staticmethod
    def _write_thumb(data: bytes, path: Path) -> None:
        img = load_image(data)
        img.thumbnail(THUMB_SIZE)
        img.save(path, "JPEG", quality=80)

    # ---- history / verification changes ---------------------------------------------------
    def get(self, obs_id: str) -> Observation:
        return self.store.get(obs_id)

    def _filters(self, status, q, species, date_from, date_to, has_location) -> dict:
        for name, v in (("date_from", date_from), ("date_to", date_to)):
            if v and not _DATE_RE.match(v):
                raise ValueError(f"{name} must look like YYYY-MM-DD")
        return dict(status=parse_status(status) if status else None, q=q, species=species,
                    date_from=date_from or None, date_to=date_to or None, has_location=has_location)

    def list(self, status=None, limit: int = 50, offset: int = 0, *, q=None, species=None, date_from=None,
             date_to=None, has_location: Optional[bool] = None):
        return self.store.list(**self._filters(status, q, species, date_from, date_to, has_location),
                               limit=limit, offset=offset)

    def list_all(self, status=None, *, q=None, species=None, date_from=None, date_to=None,
                 has_location: Optional[bool] = None):
        return self.store.list_all(**self._filters(status, q, species, date_from, date_to, has_location))

    def update(self, obs_id: str, verification_status=None, notes: Optional[str] = None, *,
               user_species=UNSET, location=UNSET) -> Observation:
        """user_species: None/'' clears the correction. location: None clears; (lat, lon, source, acc) sets."""
        if user_species is not UNSET:
            user_species = normalize_species(user_species)
        return self.store.update(
            obs_id,
            verification_status=parse_status(verification_status) if verification_status else None,
            notes=None if notes is None else notes.strip(), user_species=user_species, location=location)

    def delete(self, obs_id: str) -> None:
        self.store.delete(obs_id)
