"""Pure-numpy evaluation metrics (SRS section 13). Nothing here touches models/files."""
from __future__ import annotations

import numpy as np


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    probs, y = np.asarray(probs), np.asarray(y)
    if len(y) == 0:
        return float("nan")
    topk = np.argsort(-probs, axis=1, kind="stable")[:, :k]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def confusion_matrix(probs: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    pred = np.argmax(probs, axis=1)
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred):
        m[t, p] += 1
    return m


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10):
    """ECE on top-1 confidence. Returns (ece, per-bin rows)."""
    probs, y = np.asarray(probs), np.asarray(y)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, rows = 0.0, []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                         "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return float(ece), rows


def operating_point(probs_pos, y_pos, probs_neg, threshold: float) -> dict:
    """Behaviour at one threshold (decision rule: max prob >= threshold -> IDENTIFIED)."""
    probs_pos, y_pos = np.asarray(probs_pos), np.asarray(y_pos)
    conf_pos = probs_pos.max(axis=1)
    accepted = conf_pos >= threshold
    correct = probs_pos.argmax(axis=1) == y_pos
    n_pos = len(y_pos)
    sel_acc = float(correct[accepted].mean()) if accepted.any() else float("nan")
    out = {
        "threshold": float(threshold),
        "coverage": float(accepted.mean()) if n_pos else float("nan"),   # fraction of real-class images answered
        "selective_accuracy": sel_acc,                                    # accuracy among answered
        "true_class_rejected": int((~accepted).sum()),
    }
    if probs_neg is not None and len(probs_neg):
        rejected = np.asarray(probs_neg).max(axis=1) < threshold
        out["negative_rejection_rate"] = float(rejected.mean())
        out["negatives_rejected"] = int(rejected.sum())
        out["negatives_total"] = int(len(rejected))
    return out


def threshold_sweep(probs_pos, y_pos, probs_neg, thresholds=None) -> list[dict]:
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.991, 0.01), 2)
    return [operating_point(probs_pos, y_pos, probs_neg, t) for t in thresholds]


def pick_threshold(sweep: list[dict], min_negative_rejection: float = 0.80,
                   min_selective_accuracy: float = 0.90):
    """Highest-coverage threshold meeting both constraints; None if infeasible.

    Ties -> lower threshold is NOT preferred; the first (smallest) feasible threshold
    with max coverage wins, which keeps the system as permissive as the constraints allow.
    """
    ok = [r for r in sweep
          if r.get("negative_rejection_rate", 0.0) >= min_negative_rejection
          and r["selective_accuracy"] == r["selective_accuracy"]  # not NaN
          and r["selective_accuracy"] >= min_selective_accuracy]
    if not ok:
        return None
    best = max(r["coverage"] for r in ok)
    return next(r for r in ok if r["coverage"] == best)


def latency_stats(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=float)
    if a.size == 0:
        return {"average_ms": float("nan"), "median_ms": float("nan"), "p95_ms": float("nan"), "n": 0}
    return {"average_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "n": int(a.size)}
