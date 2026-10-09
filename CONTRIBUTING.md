# Contributing

```
pip install -r requirements.txt
python -m unittest discover -s tests -t . -v            # must stay green; CI runs it on Python 3.11 and 3.12
pip install -r requirements-browser.txt && playwright install chromium
ECOID_BROWSER_TESTS=1 python -m unittest tests.ui_browser.test_browser_flow -v      # real-browser UI tests
```

- Keep the privacy rules in `docs/privacy.md`: no network access in the app, none in the page (the Content-Security-Policy and the tests enforce it).
- Never commit datasets, model weights, observation databases or photos (`.gitignore` covers `data/`, `models/`, `reports/`, `*.onnx`, `*.pt`).
- Numbers about the model (accuracy, latency) come from `scripts/evaluate.py` / `scripts/benchmark.py` on real data — never typed in by hand (SRS section 13/21).
- Every change that touches the UI should keep `tests/ui_browser` green; look at the uploaded screenshots when layout changes.
- Photos from iNaturalist carry their own licences: see `docs/data_collection.md` and keep `ATTRIBUTION.md` with any copy of the images.

The repository has **no LICENSE file yet on purpose**: the licence is the owner's decision (also consider the licences of the model weights and of the photos you train on).
