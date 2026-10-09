# Architecture

Implements `SRS.md` section 15. Dependencies point downward only; the UI never touches the model.

```
Browser UI (app/ui/static/index.html)         capture -> result -> verify -> history
        │  JSON over HTTP, loopback only
HTTP server (app/ui/server.py)                routing, validation, size limits, Host check, CSP
        │
Application layer (app/observation/)          ObservationManager: identify -> draft -> save
        │                                     Verification: VERIFIED | REJECTED | UNCERTAIN
   ┌────┴─────────────┐
AI inference           Local storage (app/storage/)
app/ai/ (Phase 1)      SQLite + photo files under the data dir
```

## Flow
1. `POST /api/identify` (raw image bytes) → validated, run through `Identifier`, stored as a **draft**.
2. The user looks at the real plant and picks Verified / Rejected / Not sure (optional note).
3. `POST /api/observations` turns the draft into an **observation**: original photo copied byte-for-byte, thumbnail written, row inserted.
4. Abandoned drafts are deleted after 24 h (at startup) or when the user discards them.

Why drafts: SRS section 16 saves an observation only after human verification. Walking away mid-flow
therefore leaves no unverified record behind.

## Data directory (`--data-dir`, default `~/.ecoid`)
```
ecoid.db                          observations (SQLite, schema version in PRAGMA user_version)
images/<yyyy>/<mm>/<id>.<ext>     original photo, full resolution (SRS section 12)
thumbs/<id>.jpg                   320 px preview for the history list
drafts/                           photos awaiting verification
```

## Observation record (schema v2)
FR-007 fields (`id, image_path, predicted_species, confidence, alternative_predictions, verification_status,
timestamp, latitude, longitude, notes, model`) **plus** `identification_status` (`IDENTIFIED` / `LOW_CONFIDENCE`),
so history can show that a suggestion was low-confidence. Phase 4 adds `user_species` (what you say it really is; only for REJECTED), `human_label` (derived), `captured_at` (EXIF), `location_source` (`exif|device|manual`) and `location_accuracy_m`. Phase 3 databases are migrated in place on first start. `predicted_species` is always the model's top-1 label —
a suggestion, never a fact; `verification_status` is what the human decided.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the page |
| GET | `/api/model` | model information (SRS section 10) |
| POST | `/api/identify` | body = image bytes → `{draft_id, result, exif:{latitude,longitude,captured_at}}` |
| DELETE | `/api/drafts/<id>` | discard a draft |
| POST | `/api/observations` | `{draft_id, verification_status, notes?, latitude?, longitude?, location_source?, location_accuracy_m?, user_species?, use_photo_location?}` |
| GET | `/api/observations?status=&q=&species=&date_from=&date_to=&has_location=&limit=&offset=` | history, newest first |
| GET/PATCH/DELETE | `/api/observations/<id>` | read / change status, notes, `user_species`, location (`latitude`+`longitude`, or `clear_location`) / delete |
| GET | `/api/stats?<filters>` | field statistics (Wilson intervals, small-n flags) |
| GET | `/api/map?<filters>` | compact points for the map (only observations with a location) |
| GET | `/api/export?format=csv\|geojson\|json\|zip&include_location=&<filters>` | download |
| GET | `/api/basemap`, `/basemap/image` | optional offline background picture |
| GET | `/images/<id>`, `/thumbs/<id>` | photos |

`<filters>` = `status, q, species, date_from, date_to, has_location` — the same for History, Map, Stats and Export.
