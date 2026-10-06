# Continuous integration

Workflows live in `.github/workflows/` (SRS section 19).

| Workflow | Runs | Blocking? |
|---|---|---|
| `ci.yml` → `tests` | Ubuntu, Python 3.11 + 3.12: byte-compile all code, then the full unit/e2e suite with `-W error::ResourceWarning` | **yes** |
| `ci.yml` → `tests-windows` | same suite on Windows, Python 3.12 | no (`continue-on-error`) until seen green once |
| `train-smoke.yml` | synthetic dataset → check → train → ONNX export + parity → calibrate → evaluate (CPU torch) | no; manual, or when training/export scripts change |

## What `tests` guarantees
- 90+ tests: pipeline, dataset checks, metrics, storage, manager, HTTP server, offline end-to-end (all non-loopback sockets and DNS blocked), and the page's real JavaScript executed under Node.
- `CI=true` (set by GitHub) turns the Node-based UI test from "skip if no node" into a hard failure, so it can't silently disappear.
- Only `requirements.txt` is installed (numpy, Pillow, onnxruntime): the app and its tests need no torch.

## Run the same thing locally
```
python -m compileall -q app scripts tests
CI=true python -W error::ResourceWarning -m unittest discover -s tests -t . -v      # PowerShell: $env:CI="true"; python ...
```

## Caveat on `train-smoke`
It was written without being able to execute PyTorch. Its first run is the first time `train.py` / `export_model.py`
actually execute; expect to fix small version issues there, then it becomes a regression guard for the training code.
Synthetic images are coloured noise — a green run means the plumbing and ONNX parity work, not that plants are recognised.

## Protecting `main`
In GitHub: Settings → Branches → require status checks `tests (py3.11, ubuntu)` and `tests (py3.12, ubuntu)`.
