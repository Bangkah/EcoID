"""Human verification (SRS FR-006)."""
from __future__ import annotations

from enum import Enum


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"      # the user looked at the real plant and agrees
    REJECTED = "REJECTED"      # the user looked and the suggestion is wrong
    UNCERTAIN = "UNCERTAIN"    # the user cannot tell


def parse_status(value) -> VerificationStatus:
    try:
        return VerificationStatus(str(value).upper())
    except ValueError:
        raise ValueError(
            f"verification_status must be one of {[s.value for s in VerificationStatus]}, got {value!r}"
        ) from None
