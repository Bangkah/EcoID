"""What the user types as 'it was really …'. The five SRS classes are canonicalised; anything else is kept as written."""
from __future__ import annotations

from typing import Optional

from app.ai.contract.labels import COMMON_NAMES


def normalize_species(text) -> Optional[str]:
    if text is None:
        return None
    if not isinstance(text, str):
        raise ValueError("user_species must be a string")
    t = " ".join(text.split())
    if not t:
        return None
    low = t.lower()
    for sci, common in COMMON_NAMES.items():
        if low in (sci.lower(), common.lower()):
            return sci
    return t
