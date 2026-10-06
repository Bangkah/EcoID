# Collecting the real dataset from iNaturalist

Tools: `scripts/fetch_inat.py` (probe → plan → select → download), `scripts/make_contact_sheets.py`,
`scripts/apply_rejections.py`, `scripts/make_metadata_template.py`, then `scripts/check_dataset.py`.
Taxa and negative groups are configured in `config/inat_taxa.toml`.

> Status: the logic is tested against a **fake** iNaturalist (response shapes taken from the public API docs).
> It has **not** been run against the live service — the machine it was written on had no internet. That is why step 1 is a
> one-request `probe`: if a field name differs from what the code expects, you find out before anything is downloaded.

## What iNaturalist asks of you (and what the tool enforces)
From their *API Recommended Practices* and developer page:
- about **1 request/second**, around **10k requests/day**, up to **200 results per page**, a descriptive **User-Agent** → the tool rate-limits at 1.1 s, pages at 200, sends your `--contact` in the User-Agent, and stops at `--max-requests` (default 1500).
- the API is **for applications and small/medium batches, not bulk scraping**; for bulk data they point to exports and the GBIF dataset. ~2,500 images is small, but if you want much more, use those instead.
- **> 5 GB of media/hour or 24 GB/day can get you permanently blocked** → the tool downloads `medium` (~500 px, tens of KB each; the whole dataset is a few hundred MB) and stops at `--budget-gb` (default 2).
- Only photos on the **open-data bucket** with an explicit licence are used. By default `CC0`, `CC BY`, `CC BY-SA`. `CC BY-NC(-SA)` only with `--allow-nc`; `ND` and "all rights reserved" never.

## Steps
```
# 0) on a machine WITH internet
pip install -r requirements.txt

# 1) one request: verify the API still looks the way the code expects
python scripts/fetch_inat.py probe --contact you@example.com
#    every key in "observation keys present" must be True

# 2) find the place id for the region you care about (optional but recommended, see Decisions)
python scripts/fetch_inat.py plan --contact you@example.com --place Indonesia      # prints candidate ids, then exits
python scripts/fetch_inat.py plan --contact you@example.com --place-id <ID>         # real run, ~15-30 min, no images yet

# 3) READ the plan report (printed per taxon/grade): how many photos survive, why others were dropped
# 4) choose photos; split is BY PHOTOGRAPHER; shortages are printed, never hidden
python scripts/fetch_inat.py select --eval-place-id <ID>

# 5) download images + metadata + ATTRIBUTION.md  (resumable; Ctrl-C is safe)
python scripts/fetch_inat.py download --contact you@example.com

# 6) your own non-plant objects (hands, soil, desk, wall, sky ...): copy photos into data/negative/non_plant/ then
python scripts/make_metadata_template.py --own negative/non_plant
python scripts/check_dataset.py

# 7) manual label QC (mandatory per SRS)
python scripts/make_contact_sheets.py            # -> reports/contact_sheets/*.jpg ; look at ALL of them
#    write the captions (ids) of wrong images into data/metadata/reject.txt, one per line
python scripts/apply_rejections.py --dry-run && python scripts/apply_rejections.py
python scripts/fetch_inat.py select ... && python scripts/fetch_inat.py download ...   # tops up what you rejected
python scripts/check_dataset.py                  # must end with 0 errors
```
`plan` is resumable per group (`--refresh` redoes). `select` is deterministic (`--seed`). `download` skips what exists.

## Design decisions you should know about
- **Split by photographer, not by image.** Each iNaturalist user lands in exactly one of train / val / evaluation (hash of user id), so
  no one's camera, garden or style appears on both sides of the evaluation. Within a split photos are taken round-robin
  across people with caps (train ≤ 6 per person, val ≤ 2, **evaluation = 1**), one photo per observation first.
  This is stronger than the near-duplicate check, which still runs afterwards.
- **Research grade ≠ what you want.** iNaturalist's research grade requires a *wild* organism; mango, coconut, papaya, cassava and banana
  in gardens and farms are *cultivated* ("casual"). Using research grade only would train on the unusual wild/naturalised individuals.
  Default is `research,casual`; casual observations must have ≥ 1 independent agreeing identification and no disagreement.
  Grade, cultivated flag and agreements are in the candidate files so you can analyse the effect.
- **Newest first.** Pagination goes newest→oldest (closest to today's phone cameras) and takes at most 30 photos per person before selection.
- **Negatives** (SRS section 6): `lookalike_plants` (rambutan, nangka, sirsak, Citrus, **Mangifera foetida/odorata** — the hardest, same genus as mango —
  oil palm, areca, castor, Heliconia, Canna …), `non_plant` (animals/fungi from iNaturalist + **your own** object photos), `low_quality`
  (non-target photos synthetically blurred/darkened/over-exposed/cropped/over-compressed at download time, marked "[modified]" in the credits).
  Negatives are split by photographer too, into `negative` (final benchmark) and `negative_val` (threshold calibration).
- **Everything is traceable:** `data/metadata/images.csv` (licence, credit, observation link per file), `data/ATTRIBUTION.md`,
  `data/metadata/inat/downloaded.jsonl` (sha256, bytes). `data/` is git-ignored; if you ever publish images, ship the attribution with them
  (CC BY-SA also requires sharing under the same licence).

## Decisions only you can make
1. **Pisang.** Cultivated bananas are often filed under hybrids/cultivars, not the wild species *Musa acuminata*. Look at the `plan` numbers
   for it. If it is thin, add `"Musa × paradisiaca"` (see the comment in `config/inat_taxa.toml`) — but then the class means "pisang",
   not the species, and `docs/SRS.md` should say so.
2. **Where.** The app is meant for Indonesia. Training on worldwide photos is fine, but evaluate on the target domain:
   `select --eval-place-id <Indonesia id>` restricts evaluation photos to it. (Restricting *training* too is `plan --place-id`;
   fewer photos, better match.)
3. **Non-commercial licences.** Allowed with `--allow-nc`; check that is compatible with how you will share the model/dataset.
4. **Low-quality negatives:** should a *blurry photo of a real mango* count as "unknown"? The default builds low-quality negatives only from
   non-target subjects, so the model is not told either way. Change `[negatives.low_quality].taxa` if the product decision differs.

## Known limits (be honest in the report)
- Labels come from community identification: good, not perfect. Your manual QC pass is what makes them trustworthy; report how many you rejected.
- iNaturalist photos are deliberately composed (organism centred, often macro). A phone snapshot in a field is messier. **Add your own field
  photos to the evaluation set** (SRS: "koleksi sendiri"): put them in `data/evaluation/<Class>/`, then `make_metadata_template.py --own evaluation`.
- Class balance is whatever iNaturalist offers per species; check the `select` table for shortages before training.

## Troubleshooting
| Symptom | Meaning / fix |
|---|---|
| `probe` shows `False` for a key | the API changed; don't run `plan` — tell me which key |
| `no active iNaturalist taxon named …` | fix the name in `config/inat_taxa.toml` or pin `{id = …}` |
| `plan` stops with "stopped after N requests" | progress is saved per group; run `plan` again (or raise `--max-requests`) |
| HTTP 429 | the tool backs off automatically (Retry-After); if it persists, wait an hour and re-run |
| `SHORTAGES …` in `select` | not enough distinct photographers in that split; raise `--max-candidates/--max-pages` + `plan --refresh`, widen place, or lower the target |
| `stopped by budget` in `download` | by design; run `download` again later |
