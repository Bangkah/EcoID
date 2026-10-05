"""Model contract (SRS section 7). UI/storage depend ONLY on these types."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


class Status(str, Enum):
    IDENTIFIED = "IDENTIFIED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # a.k.a. UNKNOWN in the SRS


@dataclass(frozen=True)
class Candidate:
    label: str
    score: float


@dataclass(frozen=True)
class IdentificationResult:
    candidates: tuple[Candidate, ...]
    status: Status
    model: str
    threshold: float

    @property
    def top1(self) -> Candidate:
        return self.candidates[0]

    def to_dict(self) -> dict:
        return {
            "candidates": [asdict(c) for c in self.candidates],
            "status": self.status.value,
            "model": self.model,
        }
