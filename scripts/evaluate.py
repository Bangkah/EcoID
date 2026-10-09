"""Benchmark on the held-out evaluation set + negatives (SRS section 13).

  python scripts/evaluate.py --model models/ecoid.onnx
Writes reports/benchmark.json and reports/benchmark.txt.
Do NOT use this set to pick the threshold -- use scripts/calibrate_threshold.py (val sets).
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from app.ai.config import DEFAULT_CONFIG_PATH, InferenceConfig
from app.ai.data.checks import sha256_file
from app.ai.evaluation.runner import evaluate, format_report
from app.ai.inference.backend import OnnxBackend
from app.ai.inference.identifier import Identifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-dir", default="data/evaluation")
    ap.add_argument("--negative-dir", default="data/negative")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--out", default="reports")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()

    ident = Identifier(OnnxBackend(a.model, num_threads=a.threads), InferenceConfig.load(a.config))
    res = evaluate(ident, Path(a.eval_dir), Path(a.negative_dir))
    res["model_sha256"] = sha256_file(Path(a.model))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    text = format_report(res)
    (out / "benchmark.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
