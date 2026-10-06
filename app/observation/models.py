"""Observation record (SRS FR-007)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.ai.contract.result import Status
from app.observation.verification import VerificationStatus

MAX_NOTES = 2000


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
        }
