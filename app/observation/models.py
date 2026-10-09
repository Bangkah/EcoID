"""Observation record (SRS FR-007)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.ai.contract.result import Status
from app.observation.verification import VerificationStatus

MAX_NOTES = 2000
MAX_SPECIES = 120
LOCATION_SOURCES = ("exif", "device", "manual")


@dataclass(frozen=True)
class Observation:
    id: str
    image_path: str                       # relative to the data dir
    predicted_species: str                # model's top-1 label (a suggestion, not a fact)
    confidence: float                     # top-1 score
    alternative_predictions: tuple        # remaining Top-K: ({"label":..., "score":...}, ...)
    verification_status: VerificationStatus
    timestamp: str                        # ISO-8601 UTC
    model: str
    identification_status: Status         # IDENTIFIED | LOW_CONFIDENCE at the time of capture
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: str = ""
    user_species: Optional[str] = None      # what the human says it really is (only meaningful when REJECTED)
    captured_at: Optional[str] = None       # when the photo was taken (EXIF), if known; `timestamp` = when it was saved
    location_source: Optional[str] = None   # "exif" | "device" | "manual"
    location_accuracy_m: Optional[float] = None

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be within [0, 1]")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be given together")
        if self.latitude is not None:
            if not (-90.0 <= self.latitude <= 90.0):
                raise ValueError("latitude must be within [-90, 90]")
            if not (-180.0 <= self.longitude <= 180.0):
                raise ValueError("longitude must be within [-180, 180]")
        if len(self.notes) > MAX_NOTES:
            raise ValueError(f"notes must be at most {MAX_NOTES} characters")
        if self.latitude is None:
            if self.location_source is not None or self.location_accuracy_m is not None:
                raise ValueError("location_source/accuracy given without a location")
        else:
            if self.location_source not in LOCATION_SOURCES:
                raise ValueError(f"location_source must be one of {LOCATION_SOURCES}")
            if self.location_accuracy_m is not None and not (0 <= self.location_accuracy_m <= 1_000_000):
                raise ValueError("location_accuracy_m must be between 0 and 1,000,000")
        if self.user_species is not None:
            if not self.user_species.strip() or len(self.user_species) > MAX_SPECIES:
                raise ValueError(f"user_species must be 1-{MAX_SPECIES} characters")
            if self.verification_status == VerificationStatus.VERIFIED:
                raise ValueError("a VERIFIED observation agrees with the suggestion; it cannot also carry a correction")

    @staticmethod
    def from_dict(d: dict) -> "Observation":
        """Inverse of to_dict(); runs every validation. Raises ValueError/KeyError/TypeError on malformed input."""
        return Observation(
            id=str(d["id"]), image_path=str(d["image_path"]), predicted_species=str(d["predicted_species"]),
            confidence=float(d["confidence"]),
            alternative_predictions=tuple({"label": str(a["label"]), "score": float(a["score"])}
                                          for a in d.get("alternative_predictions") or ()),
            verification_status=VerificationStatus(d["verification_status"]), timestamp=str(d["timestamp"]),
            model=str(d["model"]), identification_status=Status(d["identification_status"]),
            latitude=d.get("latitude"), longitude=d.get("longitude"), notes=str(d.get("notes") or ""),
            user_species=d.get("user_species"), captured_at=d.get("captured_at"),
            location_source=d.get("location_source"), location_accuracy_m=d.get("location_accuracy_m"))

    @property
    def human_label(self) -> Optional[str]:
        """The label a human vouches for: the suggestion if VERIFIED, their correction if REJECTED, else unknown."""
        if self.verification_status == VerificationStatus.VERIFIED:
            return self.predicted_species
        if self.verification_status == VerificationStatus.REJECTED:
            return self.user_species
        return None

    def to_dict(self) -> dict:
        """FR-007 keys, plus `identification_status` (needed to show 'low confidence' in history)."""
        return {
            "id": self.id,
            "image_path": self.image_path,
            "predicted_species": self.predicted_species,
            "confidence": self.confidence,
            "alternative_predictions": [dict(a) for a in self.alternative_predictions],
            "verification_status": self.verification_status.value,
            "timestamp": self.timestamp,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "notes": self.notes,
            "model": self.model,
            "identification_status": self.identification_status.value,
            "user_species": self.user_species,
            "human_label": self.human_label,
            "captured_at": self.captured_at,
            "location_source": self.location_source,
            "location_accuracy_m": self.location_accuracy_m,
        }
