# Dataset

How to build it from iNaturalist: **`docs/data_collection.md`**.

Spec: `SRS.md` section 6. Validated by `python scripts/check_dataset.py`.

## Layout
```
data/
├── train/<Class_name>/*.jpg          300–500 per class (SRS)
├── val/<Class_name>/*.jpg            model selection + threshold calibration     [added in Phase 2]
├── evaluation/<Class_name>/*.jpg     exactly 10 per class, final benchmark only
├── negative/<group>/*.jpg            100+ total, final unknown-rejection benchmark
│   ├── lookalike_plants/             rambutan, nangka, sirsak, jeruk … (REQUIRED, per SRS)
│   ├── non_plant/
│   └── low_quality/                  blur, dark, etc.
├── negative_val/<group>/*.jpg        same groups, different images; threshold calibration [added in Phase 2]
└── metadata/images.csv               path,split,label,source,license,author,url
```
Class folder = scientific name with `_` (e.g. `Mangifera_indica`). Order is fixed by `app/ai/contract/labels.py`.

## Why two extra folders (deviation from the SRS tree)
The SRS has only `train`, `evaluation`, `negative`. If the threshold is tuned on `evaluation`/`negative` and the
benchmark is reported on the same images, the reported numbers are optimistic. So:
- `val` + `negative_val` → choose the threshold (`calibrate_threshold.py`)
- `evaluation` + `negative` → report final numbers (`evaluate.py`), run once the threshold is frozen

## What `check_dataset.py` enforces
| Check | Severity |
|---|---|
| expected class folders in train/val/evaluation; non-empty train & val | error |
| train count outside 300–500, evaluation ≠ 10 per class | warning |
| negative < 100 images; `lookalike_plants`, `non_plant`, `low_quality` missing/empty | error |
| unreadable/corrupt image | error |
| **exact or near-duplicate (dHash ≤ 5 bits) across different splits** — i.e. leakage | **error** |
| duplicates inside one split | warning |
| image without a row in `images.csv`, or empty license | error |
| non-commercial license (…-NC) | warning (check against your project license) |

Typical workflow: collect → `python scripts/make_metadata_template.py` → fill `source/license/author/url` →
`python scripts/check_dataset.py` → fix until 0 errors → manual QC of labels (the script cannot judge labels).

## Rules carried over from the SRS
- Disease datasets only as augmentation of negatives, never for main training.
- Prefer evaluation images from a different source/condition than training (e.g. your own field photos).
- Web-scraped data must be deduplicated and label-QC'd by hand.
