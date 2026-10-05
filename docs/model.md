# Model

Spec: `SRS.md` section 5.

| Item | Decision |
|---|---|
| Backbone | MobileNetV3-Small, torchvision ImageNet-1K weights (fallback: MobileNetV3-Large via `--arch large`) |
| Head | new `Linear(…, 5)` replacing the 1000-class layer; outputs raw **logits** |
| Input | `float32 [N,3,224,224]`, RGB, /255, ImageNet mean/std, **direct resize** (no crop) |
| Runtime | ONNX Runtime, CPUExecutionProvider |
| Training | transfer learning only; from-scratch training is refused by `scripts/train.py` |

## Training recipe (`scripts/train.py`)
1. **Head stage**: backbone frozen (BatchNorm kept in eval mode), AdamW lr 1e-3, 5 epochs.
2. **Fine-tune stage**: last `--unfreeze-blocks` (default 5 of 13) blocks of `model.features` + head, AdamW lr 1e-4, cosine decay, 20 epochs, early stopping (patience 6).
3. Label smoothing 0.1 (helps confidence calibration). Seeded.
4. Augmentation: rotation ±20°, random-resized-crop (scale 0.6–1, near-square), horizontal flip, brightness/contrast/saturation jitter. **No shear/perspective/elastic.**
5. Model selection by **val** accuracy only. `data/evaluation` is never touched during training.

The SRS says "unfreeze 15–30 last layers" (Keras-style layer count). In torchvision the natural unit is a
`features` block; 5 blocks of MobileNetV3-Small is roughly in that range. It is a CLI flag, so it can be tuned
against val accuracy.

## Export (`scripts/export_model.py`)
Writes `models/<name>.onnx` plus `models/<name>.onnx.json` (sha256, class order, input/output spec, training
fingerprint, library versions, parity results). The script **exits non-zero** if:
- torch vs ONNX Runtime logits differ by > 1e-3 or argmax differs, or
- torch eval-transform and `app/ai/preprocessing` differ by > 1e-4 on real images, or
- end-to-end top-1 disagrees on any sample image.

## Known limitation: unknown detection
A 5-way softmax always sums to 1, so "unknown" is decided by max-probability thresholding. This is a baseline
and is often weak against lookalike plants (rambutan, nangka, …). If the negative-rejection target cannot be met
(`calibrate_threshold.py` reports "No threshold satisfies the constraints"), options in order of cost:
1. add more/better lookalike negatives to `data/negative_val` and re-check;
2. Outlier-Exposure training (negatives with a uniform target) — keeps 5 outputs;
3. fall back to MobileNetV3-Large.
Do not weaken the targets just to get a number.
