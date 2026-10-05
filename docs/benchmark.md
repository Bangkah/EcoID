# Benchmark

Spec: `SRS.md` section 13. **No accuracy/latency numbers are recorded here until a real run exists** (SRS integrity rule).

## Procedure
```
python scripts/check_dataset.py                                   # 0 errors required
python scripts/train.py --data data --arch small --out models/checkpoints/run1
python scripts/export_model.py --checkpoint models/checkpoints/run1 --out models/ecoid.onnx --sample-dir data/val
python scripts/calibrate_threshold.py --model models/ecoid.onnx   # val sets -> recommended threshold
# edit config/inference.toml (threshold), then ONCE:
python scripts/evaluate.py --model models/ecoid.onnx              # evaluation + negative
```
Outputs: `reports/benchmark.txt` (SRS format), `reports/benchmark.json` (incl. confusion matrix, calibration
bins, model sha256, hardware), `reports/calibration.json`.

## Metrics
- Top-1 / Top-3 accuracy on `data/evaluation`.
- Latency: per-image wall time of preprocess + inference + softmax after 3 warm-up runs; average, median, p95; hardware recorded.
- Unknown rejection rate: share of `data/negative` with max-prob < threshold.
- Selective accuracy: accuracy among evaluation images with max-prob ≥ threshold; reported with **coverage** (share answered) — selective accuracy alone can look great by answering almost nothing.
- ECE (10 bins) as a calibration summary.

## Reading the numbers honestly
- 50 evaluation images ⇒ each image is 2 percentage points. Report counts (e.g. 44/50), not just percentages; differences of a few points are noise.
- 100 negatives bound rejection-rate precision similarly.
- Report the claim in the SRS section 21 form, with the exact evaluation-set size.
