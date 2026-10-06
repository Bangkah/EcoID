# Phase 3 — Application layer

| Item | Where | Status |
|---|---|---|
| 3.1 Observation storage (FR-007) | `app/observation/models.py`, `app/storage/store.py`, `app/observation/manager.py` | done, tested |
| 3.2 UI: capture → result → verify | `app/ui/static/index.html`, `app/ui/server.py` | done; logic tested, **not yet seen in a real browser** |
| 3.3 History view | same page, History tab | done; same caveat |
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

## Not verified here
No browser was available in the build environment. Not exercised: layout/CSS, `<dialog>` behaviour, the
camera inputs (`capture`), webcam (`getUserMedia`), phone use. Do a 5-minute manual pass (checklist below).

### Manual checklist
- [ ] choose a file → result card shows photo, 3 candidates with bars, common names
- [ ] a blurry/unrelated photo shows the "Low confidence" banner
- [ ] Verified / Rejected / Not sure each saves and shows "Saved ✓"
- [ ] History lists it with thumbnail; filter works; tapping opens the detail dialog; status/notes edits stick; delete asks first
- [ ] ⓘ shows model name, runtime, "Local / CPU", "5 plant species"
- [ ] webcam works (allow camera); on a phone via `--host 0.0.0.0`, "Take photo" opens the camera
- [ ] unplug the network → everything above still works
