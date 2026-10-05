"""Logits -> Top-K -> status (SRS sections 7 and 8)."""
from __future__ import annotations

import numpy as np

from app.ai.contract.result import Candidate, IdentificationResult, Status


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("Model output contains NaN/Inf")
    x = x - x.max()                      # numerical stability
    e = np.exp(x)
    return (e / e.sum()).astype(np.float64)


def build_result(
    logits: np.ndarray,
    labels: tuple[str, ...],
    *,
    threshold: float,
    model_name: str,
    top_k: int = 3,
) -> IdentificationResult:
    probs = softmax(logits)
    if probs.size != len(labels):
        raise ValueError(
            f"Model returned {probs.size} outputs, expected {len(labels)} classes"
        )
    k = min(top_k, len(labels))
    # stable ordering: highest score first, ties -> lower class index
    order = sorted(range(len(labels)), key=lambda i: (-probs[i], i))[:k]
    candidates = tuple(Candidate(labels[i], float(probs[i])) for i in order)
    status = (
        Status.IDENTIFIED if candidates[0].score >= threshold else Status.LOW_CONFIDENCE
    )
    return IdentificationResult(candidates, status, model_name, threshold)
