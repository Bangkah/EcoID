# Phase 3 — Application layer

| Item | Where | Status |
|---|---|---|
| 3.1 Observation storage (FR-007) | `app/observation/models.py`, `app/storage/store.py`, `app/observation/manager.py` | done, tested |
| 3.2 UI: capture → result → verify | `app/ui/static/index.html`, `app/ui/server.py` | done; tested in real Chromium (`tests/ui_browser/`) |
| 3.3 History view | same page, History tab | done; tested in real Chromium |
| 3.4 End-to-end offline test | `tests/ui/test_app_e2e.py`, `tests/ui/smoke.js` | done |

## Run
```
python -m app --model models/ecoid.onnx --open        # http://127.0.0.1:8765
```
Before the real model exists you can try the UI with the synthetic one:
`python scripts/make_test_onnx.py models/tiny_test.onnx` then `--model models/tiny_test.onnx`
(its "predictions" are meaningless — it only reacts to image colour).

## What the tests prove
- storage: round trip, ordering/paging/filter, update, delete (files too), DB-level CHECK constraints, persistence across restarts, schema-version guard
- manager: nothing saved before verification, original photo byte-identical (not downscaled to 224), thumbnail, model name recorded on every observation, draft safety (invalid status/coordinates keep the draft; no double save; no path traversal via draft id)
- server: error codes, 25 MB limit, Host-header check, traversal attempts, JSON-only data endpoints
- offline: full flow with the network blocked → zero outbound attempts; page contains no external references
- UI script: the page's real JavaScript is executed (stub DOM, real HTTP) through capture → verify → history → edit → delete

## Real-browser verification
`tests/ui_browser/test_browser_flow.py` runs the app in headless Chromium (Playwright) and saves screenshots.
It found and led to a fix of one real issue (history could show stale results when the filter changed quickly).
Automated: layout/overflow on a 390 px viewport, `<dialog>`, thumbnails and photos actually loading, low-confidence banner,
all three verification choices, history filter/edit/delete, model-info dialog, dark mode, webcam path with Chromium's fake
camera, zero outbound requests, zero JavaScript errors.

## Still manual (needs physical devices)
- [ ] a **real camera** (laptop webcam) — allow the permission prompt, capture, get a result
- [ ] a **phone** on the same Wi-Fi: `python -m app --model ... --host 0.0.0.0`, open the printed address, "Take photo" opens the camera
- [ ] glance at the uploaded `ui-screenshots` artifact once; automated checks cannot judge taste
- [ ] unplug the network and repeat one identify → verify cycle
