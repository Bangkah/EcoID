# Phase 4 — Field observation

Phase 3 already stores photo, suggestion, verification, notes, timestamp and history. Phase 4 adds what was still missing
for real use outdoors.

| # | Feature | Where |
|---|---|---|
| 4.1 | **Location, strictly opt-in** — from the photo's EXIF, the device's GPS button, or typed; source + accuracy recorded | `app/observation/exif.py`, result card "📍 Location" |
| 4.2 | **Capture time** from EXIF (`captured_at`), separate from the save time | same |
| 4.3 | **"What was it really?"** when you reject: one of the 5 classes or any name; becomes the human label | `app/observation/species.py`, result card + detail dialog |
| 4.4 | **Search & filters** in History: free text (notes, scientific name, *Mangga/Pisang…*, your correction), status, only-with-location; API also has species and date range | `GET /api/observations?q=&species=&date_from=&date_to=&has_location=` |
| 4.5 | **Edit** location / correction / notes after the fact | detail dialog |
| 4.6 | **Close the loop:** your own VERIFIED and corrected photos → dataset (`data/evaluation`, `val`, `train`, or negatives) | `scripts/export_observations.py` |
| — | Schema v2 with an in-place migration of Phase 3 databases | `app/storage/store.py` |

## Why these
- The SRS wants **own field photos** in the evaluation set; 4.6 turns day-to-day use into exactly that, with `source=EcoID field photo, license=own`.
- A *rejected* observation with a correction is the most valuable data the app produces: a case where the model was wrong and a human says what it was.
  A rejection without a correction, or "not sure", is never exported (the label is unknown).
- Touch-grass principle kept: nothing new is required. Location and correction are optional; the extra step exists only when you press **Rejected**.

## Rules baked in
- **Location is never attached silently.** The photo's EXIF position is shown ("This photo contains a location…") and used only if you tap
  *Use the photo's location* (or tick *Always use…*, remembered in the browser). EXIF capture **time** is stored automatically (it is not a place).
- `(0, 0)` GPS values from cameras without a fix are ignored.
- `human_label`: VERIFIED → the suggestion; REJECTED → your correction (may be empty); UNCERTAIN → none. Switching back to VERIFIED drops an old correction.
- Export never mixes places: choose **one split per garden/site** (`--split`), otherwise the same plants end up in train and evaluation.
  `check_dataset.py` still runs afterwards for duplicates.

## Verified by tests (run in CI)
- EXIF parsing (N/S/E/W, offsets, junk, no-fix), never raises.
- Migration of a hand-built Phase 3 database (this found a real bug: legacy rows with coordinates but no source made the whole history fail to load; fixed by backfilling `manual`).
- Search: case-insensitive, common names, `%`/`_`/SQL-looking input treated literally, date precedence (capture date beats save date), combined filters with correct totals.
- API validation (types, ranges, sources, drafts survive failed saves), and the whole flow with every non-loopback socket blocked.
- Export: what goes where, idempotence, per-class cap, dry run, EXIF stripping keeps orientation, dataset checker accepts the result.
- **Real Chromium** (19 tests): EXIF location opt-in, "always use" persistence, the geolocation button (with a mocked position and with permission denied),
  typed/decimal-comma/invalid coordinates, reject → correction → edit → verify, cancel/skip, search + location filter, edit/clear location in the dialog, plus all Phase 3 flows.

## Not verified (needs real hardware)
- [ ] a real phone camera's EXIF (orientation + GPS) end to end: take a photo with location services on, upload, confirm the coordinates match where you stood
- [ ] the browser's "Use my current location" on a real phone (GPS) and on your laptop (may use an online lookup — see `privacy.md`)
- [ ] glance at the screenshots artifact (`14_…` to `17_…`) when the UI changes
