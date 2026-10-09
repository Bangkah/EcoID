"""Pick the Unknown threshold on VALIDATION data (never on data/evaluation).

  python scripts/calibrate_threshold.py --model models/ecoid.onnx \
      --val-dir data/val --negative-val-dir data/negative_val \
      --min-negative-rejection 0.8 --min-selective-accuracy 0.9
Prints the sweep and a recommendation; it does NOT edit config/inference.toml.
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from app.ai.config import DEFAULT_CONFIG_PATH, InferenceConfig
from app.ai.evaluation import metrics
from app.ai.evaluation.runner import collect
from app.ai.inference.backend import OnnxBackend
from app.ai.inference.identifier import Identifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val-dir", default="data/val")
    ap.add_argument("--negative-val-dir", default="data/negative_val")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--min-negative-rejection", type=float, default=0.80)
    ap.add_argument("--min-selective-accuracy", type=float, default=0.90)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    ident = Identifier(OnnxBackend(a.model), InferenceConfig.load(a.config))
    pos = collect(ident, Path(a.val_dir), labelled=True)
    neg = collect(ident, Path(a.negative_val_dir), labelled=False)
    if not len(pos.labels) or not len(neg.labels):
        raise SystemExit("Need both val images and negative_val images to calibrate.")
    sweep = metrics.threshold_sweep(pos.probs, pos.labels, neg.probs)
    best = metrics.pick_threshold(sweep, a.min_negative_rejection, a.min_selective_accuracy)

    print(f"val images: {len(pos.labels)}   negative_val images: {len(neg.labels)}")
    print("thr   coverage  sel.acc  neg.reject")
    for r in sweep[::5]:
        print(f"{r['threshold']:.2f}  {r['coverage']:.3f}    {r['selective_accuracy']:.3f}   {r['negative_rejection_rate']:.3f}")
    if best is None:
        print("\nNo threshold satisfies the constraints. Improve the model/data or relax the targets.")
    else:
        print(f"\nRecommended threshold: {best['threshold']:.2f}  "
              f"(coverage {best['coverage']:.1%}, selective acc {best['selective_accuracy']:.1%}, "
              f"negative rejection {best['negative_rejection_rate']:.1%})")
        print("Set it in config/inference.toml, then run scripts/evaluate.py for the final numbers.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(
        {"constraints": {"min_negative_rejection": a.min_negative_rejection,
                         "min_selective_accuracy": a.min_selective_accuracy},
         "recommended": best, "sweep": sweep}, indent=2))


if __name__ == "__main__":
    main()
