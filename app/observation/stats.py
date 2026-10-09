"""Field statistics (SRS Eco Mapper). Pure functions over Observation objects.

These numbers describe YOUR field use, not the model's benchmark: you choose what to photograph and what to verify,
so the sample is self-selected. The UI says so; small samples are flagged instead of dressed up as percentages.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional

from app.observation.models import Observation
from app.observation.verification import VerificationStatus as V

CONF_BINS = [(0.0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.9), (0.9, 1.0000001)]
MIN_N = 10          # below this a percentage is shown with a "too few to conclude" flag


def wilson(k: int, n: int, z: float = 1.96) -> Optional[tuple[float, float]]:
    """95% Wilson score interval for a proportion k/n (well behaved for small n and k=0 or k=n)."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(k: int, n: int) -> dict:
    ci = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci_low": ci[0] if ci else None,
            "ci_high": ci[1] if ci else None, "too_few": n < MIN_N}


def day_of(o: Observation) -> str:
    return (o.captured_at or o.timestamp)[:10]


def compute_stats(observations: list[Observation]) -> dict:
    n = len(observations)
    by_status = Counter(o.verification_status.value for o in observations)
    decided = [o for o in observations if o.verification_status in (V.VERIFIED, V.REJECTED)]
    verified = sum(1 for o in decided if o.verification_status == V.VERIFIED)

    conf_rows = []
    for lo, hi in CONF_BINS:
        sel = [o for o in decided if lo <= o.confidence < hi]
        r = _rate(sum(1 for o in sel if o.verification_status == V.VERIFIED), len(sel))
        conf_rows.append({"label": f"{lo:.2f}–{min(hi, 1.0):.2f}", "low": lo, "high": min(hi, 1.0), **r})

    by_ident = {}
    for st in ("IDENTIFIED", "LOW_CONFIDENCE"):
        sel = [o for o in observations if o.identification_status.value == st]
        c = Counter(o.verification_status.value for o in sel)
        by_ident[st] = {"n": len(sel), "VERIFIED": c["VERIFIED"], "REJECTED": c["REJECTED"], "UNCERTAIN": c["UNCERTAIN"]}

    suggested = Counter(o.predicted_species for o in observations)
    labelled = Counter(o.human_label for o in observations if o.human_label)
    corrections = Counter((o.predicted_species, o.user_species) for o in observations
                          if o.verification_status == V.REJECTED and o.user_species)
    per_species = {}
    for sp in suggested:
        sel = [o for o in decided if o.predicted_species == sp]
        per_species[sp] = _rate(sum(1 for o in sel if o.verification_status == V.VERIFIED), len(sel))

    days = defaultdict(int)
    for o in observations:
        days[day_of(o)] += 1

    return {
        "total": n,
        "by_status": {k: by_status.get(k, 0) for k in ("VERIFIED", "REJECTED", "UNCERTAIN")},
        "with_location": sum(1 for o in observations if o.latitude is not None),
        "low_confidence": by_ident["LOW_CONFIDENCE"]["n"],
        "field_agreement": _rate(verified, len(decided)),         # VERIFIED / (VERIFIED + REJECTED)
        "by_confidence": conf_rows,
        "by_identification_status": by_ident,
        "suggested": dict(suggested.most_common()),
        "human_labelled": dict(labelled.most_common()),
        "agreement_by_species": per_species,
        "corrections": [{"suggested": a, "actually": b, "count": c} for (a, b), c in corrections.most_common(20)],
        "per_day": dict(sorted(days.items())),
        "caveat": ("These figures describe your own field use, not a benchmark: you choose what to photograph and what to "
                   "verify. Rates based on fewer than %d decisions are flagged." % MIN_N),
    }
