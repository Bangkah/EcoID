# Phase 5 — Eco Mapper (SRS section 18: "GPS, observation map, field statistics, export")

GPS capture came in Phase 4. This phase adds the rest.

| # | Feature | Where |
|---|---|---|
| 5.1 | **Offline observation map** — pan, zoom, scale bar, coordinate grid, clustering, colour by verification or species, click a marker → popup → details | **Map** tab, `GET /api/map` |
| 5.2 | **Optional offline background picture** (your own map image, georeferenced by 4 numbers) | `basemap.json` in the data folder, `app/observation/basemap.py` |
| 5.3 | **Field statistics** — totals, agreement with the suggestion (with 95% interval and a "too few" flag), confidence vs. your decisions, confident vs. low-confidence answers, what it really was when wrong, activity | **Stats** tab, `GET /api/stats`, `app/observation/stats.py` |
| 5.4 | **Export** — CSV, GeoJSON, JSON, and a full **backup ZIP** with the original photos; "include locations" switch; honours the current search/filters | **Export…** button, `GET /api/export`, `app/observation/export.py` |
| 5.5 | **Restore** a backup (checksums verified, hostile archives rejected) | `scripts/import_backup.py` |
| 5.6 | **Latency benchmark** per stage vs. NFR-002 | `scripts/benchmark.py` |
| — | One shared search/status/location filter now drives History, Map, Stats and Export | page header |

## The map is offline on purpose
SRS section 3.1/11 and the page's Content-Security-Policy forbid loading anything from the internet, and map tiles are
exactly that. So observations are drawn on a Web-Mercator coordinate grid with a scale bar; **no tile is ever requested**.
For a real background, either
1. **export GeoJSON** and open it in QGIS / geojson.io / Google Earth Pro (they bring their own maps), or
2. give the app a picture of your area. Put the image in the data folder (`~/.ecoid` by default) and create `basemap.json`:
   ```json
   {"image": "kebun.png", "bounds": [5.15, 97.10, 5.25, 97.20], "attribution": "© whoever made the picture"}
   ```
   `bounds` = `[south, west, north, east]` of the picture. It must be a **Web-Mercator** image (what screenshots/exports of
   OpenStreetMap-style maps are) — an image from another projection will be misaligned. `.png/.jpg/.webp`, ≤ 20 MB, inside the data folder;
   anything invalid is ignored and the reason is shown under the map. Respect the picture's licence.

## About the statistics (what they are and are not)
- They describe **your field use**, not the model's benchmark: you choose what to photograph and what to verify, so the sample is self-selected
  (people tend to verify easy cases). Use `docs/benchmark.md` numbers for claims about the model.
- *Agreement* = VERIFIED ÷ (VERIFIED + REJECTED). "Not sure" is excluded, and shown separately.
- Rates come with a 95% **Wilson** interval, and anything based on fewer than **10** decisions is flagged "too few to conclude".
- The confidence table is a field-side reliability check: agreement should rise with the model's confidence. If high-confidence suggestions are often
  rejected, recalibrate (`scripts/calibrate_threshold.py`) or look at the "what it really was" list for systematic confusions.

## Export details
- **CSV**: UTF-8 with BOM (Excel-friendly); cells that begin with `= + - @` get a leading `'` so a spreadsheet won't execute them as formulas.
- **GeoJSON**: RFC 7946, coordinates `[longitude, latitude]`, only observations with a location (so it refuses "without locations").
- **Backup ZIP**: `manifest.json` (SHA-256 per photo), `observations.json`, `images/<id>.<ext>` — the **original** photos, including their EXIF.
  Restore never trusts names inside the archive and skips ids that already exist.
- *Include locations = off* blanks latitude/longitude/source/accuracy in CSV/JSON/ZIP. It does **not** strip EXIF GPS from photos inside a ZIP.

## Verified by tests (CI)
- Stats maths (Wilson against known values, bins, corrections), CSV injection guard, GeoJSON axis order, no-location exports leak nothing,
  backup round-trip into a fresh app, corrupted-photo and zip-slip archives rejected, basemap validation, endpoint validation and filters,
  everything with every non-loopback socket blocked.
- **Real Chromium** (25 browser tests in total, 6 new for this phase): markers/colours/legend/scale bar/popup→details/zoom/pan/wheel,
  species colouring, clustering → zoom → list at one spot, empty state, **basemap georeferencing** (marker lands inside the picture), stats
  (67% = 8 of 12, correction table, small-n flag, empty state), and real downloads of CSV / GeoJSON / ZIP with the privacy switches.
  Every test also fails on any JavaScript error or any request leaving the app's origin.
- Found by the browser tests and fixed: the page header overflowed on a 390 px phone once there were four tabs; the map popup was clipped by its
  own image loading late; cluster zoom stopped one level short for points in a straight line.

## Not verified here
- [ ] the map with *your* real observations and *your* basemap image (alignment depends on the image you provide)
- [ ] scrolling/pinching the map on a real phone touch screen (mouse wheel/drag are tested; touch uses the same pointer events)
- [ ] the backup restore on a different machine / OS
