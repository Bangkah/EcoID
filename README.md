# EcoID — Offline AI Field Mapper

Offline-first plant identification (5 classes) with local ONNX inference.
EcoID generates identification candidates; the user verifies them.

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # Python >= 3.11
```

## Test
```
python -m unittest discover -s tests -t . -v
```
CI runs lint, the suite on Python 3.11/3.12 and real-browser (Chromium) UI tests on every push/PR (`docs/ci.md`).

## Try it
```
python scripts/identify.py photo.jpg --model models/<model>.onnx
```

## Configuration
`config/inference.toml` is the single source for `model_name`, `threshold`, `top_k`.
The threshold is an initial value; it is calibrated in Phase 2.

## Phase 2 (model/data quality)
Real data: `docs/data_collection.md` (`scripts/fetch_inat.py probe|plan|select|download`).
```
pip install -r requirements-train.txt        # only for training/export
python scripts/check_dataset.py              # layout, leakage, licenses
python scripts/train.py --data data --out models/checkpoints/run1
python scripts/export_model.py --checkpoint models/checkpoints/run1 --out models/ecoid.onnx --sample-dir data/val
python scripts/calibrate_threshold.py --model models/ecoid.onnx
python scripts/evaluate.py --model models/ecoid.onnx
python scripts/benchmark.py --model models/ecoid.onnx        # latency vs NFR-002
python scripts/summarize_run.py                              # one paste-able summary of everything above
```
See `docs/dataset.md`, `docs/model.md`, `docs/benchmark.md`.

## Run the app (Phase 3)
```
python -m app --model models/ecoid.onnx --open      # then use the browser at http://127.0.0.1:8765
```
Photos and observations are stored in `~/.ecoid` (change with `--data-dir`). No internet is used.

## Docs
`docs/SRS.md` requirements · `docs/phase4.md` (field observation) · `docs/phase5.md` (offline map, statistics, export/backup) · **`docs/srs_compliance.md`** (what is met, partly met, not met) · **`docs/first_run.md`** (runbook to the first real model) · `docs/architecture.md` · `docs/privacy.md` · per-phase notes in `docs/phase*.md`.

## Status
Phase 1 done. Phase 2 now has a provisional global iNaturalist dataset and trained MobileNetV3-Small ONNX model; the model reached 80.3% validation accuracy and passed PyTorch/ONNX parity checks. Phases 3-5 (app, field observation, Eco Mapper) are done; the SRS audit is in `docs/srs_compliance.md`. Final field evaluation remains pending because the Indonesia evaluation split is still too small.

Turn your field photos into dataset files: `python scripts/export_observations.py --split evaluation --per-class 10`.
