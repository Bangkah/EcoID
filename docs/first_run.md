# First real run — runbook (pilot → first trained model)

No new features: this is the order of work that turns the finished code into the first real numbers. Do the steps in order;
each has a **stop condition** — if it fires, fix that before going on.

## 0. Prove the code, before spending hours on data (cheap, can run today)
- Push the repo → check that `lint`, `tests` (3.11, 3.12) and `ui-e2e` are green.
- Actions → **train-smoke** → *Run workflow*. This is the **first time `train.py` / `export_model.py` ever execute**.
  Synthetic images, 2 epochs, but the whole chain runs: check → train → ONNX export → parity → calibrate → evaluate.
  *Stop if red:* send the log; do not collect data against code that has never run.
- On a machine with internet: `python scripts/fetch_inat.py probe --contact you@example.com`.
  *Stop if any key is `False`.*

## 1. Pilot dataset (≈ 1–2 evenings, mostly waiting)
```
python scripts/fetch_inat.py plan --contact you@example.com --place Indonesia          # note the id, then:
python scripts/fetch_inat.py plan --contact you@example.com --place-id <ID> --max-candidates 800 --max-pages 8
#   READ the report. Pisang thin? decide about "Musa × paradisiaca" (docs/data_collection.md) and `plan --refresh`.
python scripts/fetch_inat.py select --train 80 --val 20 --eval 10 --eval-place-id <ID>
#   Keep the default negatives (110): check_dataset.py REQUIRES >= 100 negatives, so "20 negatives" would fail the checker.
python scripts/fetch_inat.py download --contact you@example.com
```
Add your own photos: **non-plant objects** → `data/negative/non_plant/` (`make_metadata_template.py --own negative/non_plant`),
and — most valuable — **your own field photos of the 5 species** into `data/evaluation/<Class>/` (`--own evaluation`).
Then QC: `make_contact_sheets.py` → look at every sheet → list wrong ids in `data/metadata/reject.txt` → `apply_rejections.py` → `select`+`download` again to top up.
`python scripts/check_dataset.py` → **0 errors** (count *warnings* are expected for a pilot: SRS wants 300–500/class).
*Stop if:* errors about duplicates/leakage between splits (never "fix" by deleting the check), or a class has < 40 train photos.

## 2. First training (CPU is fine for a pilot)
```
pip install -r requirements-train.txt
python scripts/train.py --data data --arch small --out models/checkpoints/run1 --batch-size 16
```
Read the per-epoch JSON lines:
| You see | Meaning |
|---|---|
| val acc climbs, train acc a bit higher | normal |
| train ≈ 100 %, val far lower | too little / too uniform data → more photographers, not more epochs |
| val ≈ 20 % (chance for 5 classes) | a bug: labels, class order or preprocessing — stop, don't tune |
| val ≈ 100 % after 1–2 epochs | suspect leakage (look at `check_dataset.py`) or very easy data |

## 3. Export and prove ONNX = PyTorch
```
python scripts/export_model.py --checkpoint models/checkpoints/run1 --out models/ecoid.onnx --sample-dir data/val
```
*Stop if it exits non-zero:* the parity check found a PyTorch/ONNX or preprocessing mismatch. Nothing downstream is trustworthy until it passes.

## 4. Threshold — on VALIDATION data only
```
python scripts/calibrate_threshold.py --model models/ecoid.onnx
```
Pick the threshold from `negative_val` + `val`; put it in `config/inference.toml`. If it says *no threshold satisfies the constraints*,
the max-probability rule is not enough for your lookalikes — see "Known limitation" in `docs/model.md` (more lookalike negatives first).
**Never** choose it by looking at `data/evaluation`.

## 5. Benchmark — once
```
python scripts/evaluate.py  --model models/ecoid.onnx
python scripts/benchmark.py --model models/ecoid.onnx --images data/evaluation
```
Open `reports/benchmark.txt` and `reports/benchmark.json` (confusion matrix, reliability bins).
If you change *anything* after seeing these numbers (data, threshold, hyper-parameters), the evaluation set is spent: collect a new one.
Pilot caveat: 50 evaluation images ⇒ one image = 2 points; report counts (e.g. 44/50), and call it a pilot.

## 6. Field test (the evidence that matters)
Take **new** photos in a **different place/day** than anything in the dataset: ≥ 10 per species, plus ≥ 10 plants that are *not* the five
(rambutan, nangka, jeruk, grass, a hand…). Run the app, look at the real plant, Verify / Reject (+ what it was) / Not sure, with a location.
Then the **Stats** tab: agreement with 95 % interval, agreement by confidence, what it confuses. Export a backup ZIP. Don't tune on these;
if you retrain with them, they stop being a test.

## 7. Send the results for review
```
python scripts/summarize_run.py            # -> reports/run_summary.md
```
Paste that file. It holds only counts and numbers: dataset structure per split/class, photographers per split, `check_dataset` verdict, the training log, the
export parity result, calibration and benchmark (with confusion matrix), and automatic red-flag hints. No photos, file names, photographer names or coordinates.
If a step failed, paste its error output as well.

## 8. What you may write afterwards (SRS section 21)
> "On our N-image evaluation set, EcoID achieved X/N top-1 and Y/N top-3, rejected Z of M negative samples, with median latency L ms on <CPU>.
> Pilot data: ~80 training photos per species from iNaturalist (CC licences) plus own field photos."

Nothing stronger until the 300–500/class dataset exists.
