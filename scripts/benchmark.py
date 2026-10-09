"""Latency benchmark (SRS section 13 'Inference Latency' + NFR-002: <= 10 s per image).

  python scripts/benchmark.py --model models/ecoid.onnx                 # synthetic 1280x960 photos
  python scripts/benchmark.py --model models/ecoid.onnx --images data/evaluation

Reports average / median / p95 / max per stage (decode+resize, ONNX inference, softmax+top-k) and end to end, plus the
hardware. Numbers are measured, never invented; run it on the machine you care about. Writes reports/latency.{json,txt}.
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import _common  # noqa: F401
from app.ai.config import DEFAULT_CONFIG_PATH, InferenceConfig
from app.ai.data.checks import list_images
from app.ai.evaluation.metrics import latency_stats
from app.ai.evaluation.runner import hardware_info
from app.ai.inference.backend import OnnxBackend
from app.ai.inference.postprocess import build_result
from app.ai.contract.labels import CLASSES
from app.ai.preprocessing.preprocess import preprocess


def synthetic_photos(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(min(n, 8)):                          # a few distinct images, reused: enough for timing
        a = np.clip(rng.normal(120, 60, (960, 1280, 3)), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=90)
        out.append(buf.getvalue())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default=None, help="folder of real photos (default: synthetic)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--target-s", type=float, default=10.0, help="NFR-002 target per image")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    cfg = InferenceConfig.load(a.config)
    backend = OnnxBackend(a.model, num_threads=a.threads)
    if a.images:
        photos = [p.read_bytes() for p in list_images(Path(a.images))[: a.n]]
        if not photos:
            raise SystemExit(f"no images found in {a.images}")
        source = f"{len(photos)} real images from {a.images}"
    else:
        photos = synthetic_photos(a.n)
        source = "synthetic 1280x960 JPEGs (decode cost differs from real photos: prefer --images)"

    def one(data):
        t0 = time.perf_counter()
        x = preprocess(data)
        t1 = time.perf_counter()
        logits = backend.run(x)
        t2 = time.perf_counter()
        build_result(logits, CLASSES, threshold=cfg.threshold, model_name=cfg.model_name, top_k=cfg.top_k)
        t3 = time.perf_counter()
        return [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3]

    for i in range(a.warmup):
        one(photos[i % len(photos)])
    rows = [one(photos[i % len(photos)]) for i in range(a.n)]
    cols = list(zip(*rows))
    res = {"model": str(a.model), "source": source, "runs": a.n, "warmup": a.warmup, "threads": a.threads,
           "hardware": hardware_info(), "target_s": a.target_s,
           "preprocess_ms": latency_stats(cols[0]), "inference_ms": latency_stats(cols[1]),
           "postprocess_ms": latency_stats(cols[2]), "total_ms": latency_stats(cols[3])}
    res["meets_target"] = res["total_ms"]["p95_ms"] / 1000.0 <= a.target_s
    hw = res["hardware"]
    lines = [f"EcoID latency benchmark ({source})", f"runs={a.n} warmup={a.warmup} threads={a.threads or 'default'}",
             f"hardware: {hw['processor']} | {hw['cpu_count']} logical CPUs | {hw['platform']}", ""]
    for k, label in (("preprocess_ms", "decode+resize"), ("inference_ms", "ONNX inference"),
                     ("postprocess_ms", "softmax+top-k"), ("total_ms", "TOTAL")):
        s = res[k]
        lines.append(f"{label:<15} avg {s['average_ms']:8.1f} ms   median {s['median_ms']:8.1f} ms   p95 {s['p95_ms']:8.1f} ms")
    lines += ["", f"NFR-002 (p95 <= {a.target_s:.0f} s per image): {'MET' if res['meets_target'] else 'NOT MET'}"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "latency.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
