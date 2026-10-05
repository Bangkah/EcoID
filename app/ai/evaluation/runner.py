"""Run the REAL pipeline (preprocess -> ONNX -> softmax) over labelled folders."""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.ai.contract.labels import CLASSES
from app.ai.data.checks import CLASS_DIRS, list_images
from app.ai.evaluation import metrics
from app.ai.inference.identifier import Identifier
from app.ai.inference.postprocess import softmax
from app.ai.contract.result import Status
from app.ai.inference.postprocess import build_result


@dataclass
class Collected:
    probs: np.ndarray          # (N, 5)
    labels: np.ndarray         # (N,) class index; -1 for negatives
    paths: list[str]
    latency_ms: list[float]


def collect(identifier: Identifier, root: Path | None, *, labelled: bool,
            warmup: int = 3, repeat_timing: int = 1) -> Collected:
    """labelled=True: root/<Class_name>/*.jpg ; False: any images under root (negatives)."""
    items: list[tuple[Path, int]] = []
    if root is not None and Path(root).is_dir():
        if labelled:
            for idx, d in enumerate(CLASS_DIRS):
                items += [(p, idx) for p in list_images(Path(root) / d)]
        else:
            items = [(p, -1) for p in list_images(Path(root))]
    if not items:
        return Collected(np.zeros((0, len(CLASSES))), np.zeros(0, dtype=int), [], [])

    for p, _ in items[:warmup]:
        identifier.logits(p)  # warm-up so first-run overhead isn't counted

    probs, lat = [], []
    for p, _ in items:
        t0 = time.perf_counter()
        logits = identifier.logits(p)
        ps = softmax(logits)
        lat.append((time.perf_counter() - t0) * 1000.0)
        probs.append(ps)
    return Collected(np.vstack(probs), np.array([y for _, y in items]),
                     [str(p) for p, _ in items], lat)


def hardware_info() -> dict:
    import os
    import onnxruntime
    return {"platform": platform.platform(), "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(), "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__}


def evaluate(identifier: Identifier, eval_dir: Path, negative_dir: Path | None) -> dict:
    pos = collect(identifier, eval_dir, labelled=True)
    neg = collect(identifier, negative_dir, labelled=False)
    cfg = identifier.config
    n = len(pos.labels)
    res: dict = {"model_name": cfg.model_name, "threshold": cfg.threshold,
                 "eval_samples": n, "negative_samples": len(neg.labels),
                 "hardware": hardware_info()}
    if n:
        res["top1_accuracy"] = metrics.topk_accuracy(pos.probs, pos.labels, 1)
        res["top3_accuracy"] = metrics.topk_accuracy(pos.probs, pos.labels, 3)
        ece, bins = metrics.expected_calibration_error(pos.probs, pos.labels)
        res["ece"], res["calibration_bins"] = ece, bins
        res["confusion_matrix"] = metrics.confusion_matrix(pos.probs, pos.labels, len(CLASSES)).tolist()
        res["operating_point"] = metrics.operating_point(
            pos.probs, pos.labels, neg.probs if len(neg.labels) else None, cfg.threshold)
        # consistency guard: decision via the production code path must match the metric
        statuses = [build_result(np.log(np.clip(p, 1e-12, None)), identifier._labels,
                                 threshold=cfg.threshold, model_name=cfg.model_name).status
                    for p in pos.probs]
        res["identified_count"] = int(sum(s == Status.IDENTIFIED for s in statuses))
    res["latency"] = metrics.latency_stats(pos.latency_ms + neg.latency_ms)
    return res


def format_report(res: dict) -> str:
    """SRS section 13 report format. Every number comes from `res`; nothing is pre-filled."""
    def pct(x):
        return "n/a" if x is None or x != x else f"{x * 100:.1f}%"
    op = res.get("operating_point", {})
    lat = res["latency"]
    hw = res["hardware"]
    lines = [
        res["model_name"], "=" * len(res["model_name"]), "",
        f"Evaluation samples: {res['eval_samples']}", "",
        f"Top-1 Accuracy: {pct(res.get('top1_accuracy'))}",
        f"Top-3 Accuracy: {pct(res.get('top3_accuracy'))}", "",
        f"Average inference latency: {lat['average_ms']:.1f} ms",
        f"Median inference latency: {lat['median_ms']:.1f} ms",
        f"p95 inference latency: {lat['p95_ms']:.1f} ms  (preprocess + inference + softmax, n={lat['n']})", "",
        "Unknown detection:",
        f"  Negative samples: {res['negative_samples']}",
        f"  Correctly rejected: {op.get('negatives_rejected', 'n/a')}"
        + (f" ({pct(op.get('negative_rejection_rate'))})" if 'negative_rejection_rate' in op else ""), "",
        f"Selective accuracy (above threshold {res['threshold']}): {pct(op.get('selective_accuracy'))}",
        f"Coverage (real-class images answered): {pct(op.get('coverage'))}",
        f"Expected calibration error (ECE): {res.get('ece', float('nan')):.3f}", "",
        f"Hardware: {hw['processor']} | {hw['cpu_count']} logical CPUs | {hw['platform']}",
        f"Runtime: Python {hw['python']}, onnxruntime {hw['onnxruntime']}",
    ]
    if res.get("model_sha256"):
        lines.append(f"Model file sha256: {res['model_sha256']}")
    return "\n".join(lines) + "\n"
