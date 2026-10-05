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

## Try it
```
python scripts/identify.py photo.jpg --model models/<model>.onnx
```

## Configuration
`config/inference.toml` is the single source for `model_name`, `threshold`, `top_k`.
The threshold is an initial value; it is calibrated in Phase 2.

## Phase 2 (model/data quality)
```
pip install -r requirements-train.txt        # only for training/export
python scripts/check_dataset.py              # layout, leakage, licenses
python scripts/train.py --data data --out models/checkpoints/run1
python scripts/export_model.py --checkpoint models/checkpoints/run1 --out models/ecoid.onnx --sample-dir data/val
python scripts/calibrate_threshold.py --model models/ecoid.onnx
python scripts/evaluate.py --model models/ecoid.onnx
```
See `docs/dataset.md`, `docs/model.md`, `docs/benchmark.md`.

## Docs
`docs/SRS.md` is the requirements specification.

## Status
Phase 1 done (`docs/phase1.md`). Phase 2 tooling done; needs real data + training run (`docs/phase2.md`).
