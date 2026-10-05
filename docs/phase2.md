# Phase 2 — Model / data quality

Goal: everything needed to turn data into a trustworthy, benchmarked ONNX model — and to catch the classic
failure modes (leakage, train/serve skew, threshold tuned on the test set) mechanically.

## Verified in this repo (no GPU, no real data, no torch needed)
`python -m unittest discover -s tests -t . -v` — dataset checks (leakage, duplicates, licenses, negatives),
metrics (Top-K, selective accuracy, ECE, threshold sweep/pick), and an end-to-end evaluate + calibrate run
through real ONNX Runtime using the synthetic model.

## Written but NOT executed here (needs torch/torchvision/onnx + your data)
`scripts/train.py`, `scripts/export_model.py`, `scripts/_torch_common.py` — syntax-checked only. Their built-in
safety net is the export parity check; run it first and read its output. Expect to fix small API/version issues
on the first run.

## Definition of done for Phase 2
- [ ] `check_dataset.py` → 0 errors on real data
- [ ] trained model exported, parity checks pass, `.onnx.json` sidecar written
- [ ] threshold calibrated on val sets (or the "infeasible" outcome documented and addressed)
- [ ] `evaluate.py` run once → `reports/benchmark.txt` filled from real data
- [ ] decision recorded: Small is enough, or fall back to Large (with numbers)
