# Phase 1 — Pipeline correctness

Goal: prove Image -> Preprocess -> ONNX (CPU) -> Top-K -> Unknown decision works
and is deterministic, BEFORE any real model/data (Phase 2).

Run: `python -m unittest discover -s tests -t . -v`
CLI: `python scripts/identify.py photo.jpg --model models/<model>.onnx`

## Contracts that Phase 2 must respect
1. Model outputs raw LOGITS, shape (1,5) — softmax is applied in post-processing.
2. Class order = `app/ai/contract/labels.py::CLASSES`.
3. Training preprocessing = `preprocess.py` (RGB, bilinear resize to 224x224, ImageNet mean/std).
   Direct resize (no center-crop) follows SRS FR-002; train and serve must match.
4. Threshold 0.65 is inclusive (>=) and only an initial value.

## Not covered (belongs to later phases)
Accuracy, calibration, latency numbers, real plant model, storage, UI.
`models/tiny_test.onnx` is a synthetic 3x5 linear model for testing the ONNX path only.
