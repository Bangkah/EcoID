# Continuous integration

Workflows live in `.github/workflows/` (SRS section 19).

| Job | What it runs | Blocking? |
|---|---|---|
| `lint` | `ruff check` with a deliberately small rule set (syntax errors, undefined names, invalid comparisons, unused imports) | yes |
| `tests (py3.11 / py3.12, ubuntu)` | byte-compile everything, then the full suite with `-W error::ResourceWarning` | **yes** |
| `ui-e2e (chromium)` | the same app in a **real headless Chromium** via Playwright; uploads screenshots of every state | **yes** |
| `tests-windows` | the suite on Windows / Python 3.12 | no (`continue-on-error`) until seen green once |
| `train-smoke` (separate workflow) | synthetic data → check → train → ONNX export + parity → calibrate → evaluate (CPU torch) | no; manual, weekly, and on changes to training/export/preprocessing code |
| coverage (step in `tests`, py3.12) | coverage of `app/` in the job summary + HTML artifact | report only |

## What the blocking jobs guarantee
- ~100 unit/integration tests: pipeline, dataset checks, metrics, storage, manager, HTTP server, and an offline end-to-end run with
  every non-loopback socket and DNS lookup blocked (any attempt fails the test).
- `ui-e2e` (25 tests) drives Chromium through: capture → result → verify → history → edit → delete (with the `confirm()` prompt), photo/GPS/typed location, reject-and-correct, search filters, the offline map (zoom/pan/cluster/basemap), statistics, real CSV/GeoJSON/ZIP downloads, the
  low-confidence banner, each verification choice + history filter, discard, unreadable file, model-info dialog, a 390 px phone
  viewport (no horizontal scroll, tappable buttons, dialog fits), light/dark colour schemes, and Chromium's synthetic camera
  (`getUserMedia`). Every test also fails on **any JavaScript error** or **any request leaving the app's own origin**.
- `CI=true` (GitHub sets it) makes the Node-based UI-script test fail instead of skip when Node is missing.
- The browser tests are opt-in locally (`ECOID_BROWSER_TESTS=1`); in the plain `tests` job they are reported as skipped.

## Not covered by CI (stays manual)
A physical camera, a real phone (`--host 0.0.0.0` + "Take photo"), and anyone's eyes on the screenshots (they are uploaded
as the `ui-screenshots` artifact — look at them when the UI changes).

## Run locally
```
python -m compileall -q app scripts tests
CI=true python -W error::ResourceWarning -m unittest discover -s tests -t . -v      # PowerShell: $env:CI="true"; python ...

pip install -r requirements-browser.txt && playwright install chromium
ECOID_BROWSER_TESTS=1 python -m unittest tests.ui_browser.test_browser_flow -v     # screenshots -> reports/screenshots
```

## Notes
- The page's Content-Security-Policy forbids `unsafe-eval`, so the browser tests never use `page.wait_for_function(<string>)`
  (it evaluates code in the page); they poll with `page.evaluate` instead (`wait_js`).
- `train-smoke` has not been executed yet (no PyTorch where it was written). Its first run is the first real execution of
  `train.py` / `export_model.py`; expect to fix small version issues, then it becomes the regression guard. Keep it out of
  *required* checks: a required check with a path filter stays "pending" forever when it does not trigger.
- `tests-windows`: when it is green once, delete its `continue-on-error: true` line and require it.

## Protecting `main`
Settings → Branches → require: `lint (ruff)`, `tests (py3.11, ubuntu)`, `tests (py3.12, ubuntu)`, `ui-e2e (chromium)`.
