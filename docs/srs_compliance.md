# SRS compliance audit

Checked against `docs/SRS.md`. Status: ✅ met and tested · ⚠️ partly · ❌ not met yet. "Evidence" points at tests or files.
**Last audit: after Phase 5 (Eco Mapper).** The remaining ❌ all need two things that only you can supply: real photos and a real training run.

## Section 20 — Acceptance criteria (18)
| # | Criterion | Status | Evidence / what is missing |
|---|---|---|---|
| 1 | open-weight model chosen and documented | ✅ | `docs/model.md` (MobileNetV3-Small, Large fallback, ImageNet pretrained) |
| 2 | model runs / exports via ONNX Runtime | ⚠️ | runtime path tested with a synthetic ONNX (`tests/ai/test_pipeline.py`); `scripts/export_model.py` (with parity checks) **never executed** — needs PyTorch + a trained model; CI job `train-smoke` will be its first run |
| 3 | five classes can be predicted | ❌ | needs the trained model |
| 4 | inference on CPU | ✅ | `OnnxBackend` uses `CPUExecutionProvider` only |
| 5 | Top-K output | ✅ | top-3 in every result (`tests/ai/test_pipeline.py`) |
| 6 | confidence available | ✅ | per-candidate score, shown as bars |
| 7 | Unknown threshold | ✅ | `config/inference.toml` (0.65, inclusive), boundary tested |
| 8 | user can verify | ✅ | Verified / Rejected / Not sure; browser tests |
| 9 | observations stored locally | ✅ | SQLite + original photos; persistence test |
| 10 | inference works without internet | ✅ | whole flow with all non-loopback sockets/DNS blocked (`tests/ui/test_app_e2e.py`) |
| 11 | evaluation set, 50 labelled images | ❌ | tooling ready (`docs/data_collection.md`, `check_dataset.py`); photos not collected |
| 12 | negative set, 100+ images | ❌ | same; lookalike/non-plant/low-quality groups are enforced by the checker |
| 13 | Top-1 benchmark | ❌ | `scripts/evaluate.py` ready and tested on synthetic data; **no real number exists, and none is written anywhere** |
| 14 | Top-3 benchmark | ❌ | same |
| 15 | inference latency | ⚠️ | `scripts/benchmark.py` (per stage, avg/median/p95, vs NFR-002) ready; measured only with the synthetic 15-parameter model, which says nothing about MobileNetV3 |
| 16 | unknown / negative evaluation | ❌ | metrics implemented (rejection rate, selective accuracy + coverage, ECE, threshold sweep on *validation* data); needs negatives |
| 17 | model name recorded on every observation | ✅ | `Observation.model`, from config; test in `tests/observation/test_manager.py` |
| 18 | real demo with a physical plant | ❌ | needs the model, and you holding a phone next to a mango tree |

**9 ✅ · 2 ⚠️ · 7 ❌.**

## Functional requirements (section 9)
| FR | Status | Notes |
|---|---|---|
| FR-001 image input from camera / local file | ✅ | file picker, phone camera input, webcam (tested with Chromium's fake camera); a *physical* camera is a manual check |
| FR-002 preprocessing 224×224, normalise, tensor | ✅ | `app/ai/preprocessing`; train/serve parity is verified by `export_model.py` (not yet run) |
| FR-003 local inference via ONNX Runtime, no cloud | ✅ | |
| FR-004 Top-1 / Top-3 | ✅ | |
| FR-005 Unknown handling | ✅ | "Low confidence" banner uses the SRS wording |
| FR-006 manual verification | ✅ | |
| FR-007 observation storage | ✅ | all listed fields + extras (`identification_status`, `user_species`, `human_label`, `captured_at`, `location_source`, `location_accuracy_m`); schema v2 with migration |

## Other sections
| Section | Status | Notes |
|---|---|---|
| 2 scope: 5 classes + UNKNOWN | ✅ | `app/ai/contract/labels.py` |
| 3 principles | ✅ | local-first, human-in-the-loop, uncertainty-aware, "Possible identification" wording, short flow |
| 4 hardware | ⚠️ | CPU-only ✅; RAM ≥ 4 GB / storage ≥ 1 GB are not measured |
| 5 model + training strategy | ⚠️ | decided and coded (two-stage fine-tune, no shear, scratch training refused); **training code never executed** |
| 6 dataset strategy | ⚠️ | iNaturalist collector, licence tracking, split by photographer, leakage + duplicate checks, manual-QC tools; **no data collected yet** |
| 7 model contract | ✅ | `IdentificationResult`; UI depends only on it |
| 8 unknown detection | ✅ / ⚠️ | threshold + rule ✅; **calibration on real data pending** (SRS: 0.65 is only a starting value) |
| 10 model information | ✅ | ⓘ dialog: model, runtime, "Local / CPU", "5 plant species", file hash |
| 11 offline behaviour | ✅ | wording matches the SRS ("offline inference after model installation") |
| 12 local storage | ✅ | original photo kept byte-identical, separate from the 224 px tensor |
| 13 benchmark | ⚠️ | all metrics + the SRS report format implemented; numbers wait for real data |
| NFR-001 offline | ✅ | |
| NFR-002 ≤ 10 s / image | ⚠️ | tool ready, real measurement pending (a MobileNetV3-Small on a modern CPU is expected to be far below 10 s, but that is an expectation, not a result) |
| NFR-003 privacy | ✅ | no outbound connections (tested), CSP, loopback-only by default, location opt-in, `docs/privacy.md` |
| NFR-004 reproducibility | ⚠️ | pinned runtime deps, seeds, dataset fingerprint, model SHA-256 + sidecar, per-image licence/attribution; completes with the first real run |
| NFR-005 modularity | ✅ | `ModelBackend` protocol; backend swap tested |
| 15–16 architecture / pipeline | ✅ | `docs/architecture.md` |
| 17 out of scope | ✅ | nothing out-of-scope was built (no cloud inference, disease diagnosis, marketplace, RAG, chatbot…) |
| 21 claims | ✅ | no accuracy claim anywhere; statistics screens say they are not a benchmark |

## Section 18 — roadmap
| SRS phase | Status |
|---|---|
| Core identification | ✅ (pipeline); the real model is pending |
| Field observation | ✅ (photo storage, verification, notes, timestamp, history, search, corrections) |
| **Eco Mapper** (GPS, observation map, statistics, export) | ✅ — offline map; GPS from EXIF / device / typed |
| Micro-Farm Companion (garden tracking, plant care, growth monitoring) | ❌ not started |
| Community Science (shared observations, dataset contribution, collaborative verification) | ❌ not started — and it conflicts with the SRS's own out-of-scope list ("social network", "collaborative database"), so it needs a decision first |

Note: this repository numbers its build phases differently (1 pipeline, 2 model/data, 3 application, 4 field observation, 5 Eco Mapper); see `docs/SRS.md` section 18.

## Section 19 — repository structure
| Item | Status |
|---|---|
| README.md, CONTRIBUTING.md | ✅ |
| **LICENSE** | ❌ deliberately absent: choosing a licence is the owner's decision |
| docs/architecture, model, dataset, benchmark, privacy | ✅ (plus SRS, data_collection, ci, phase*.md, this file) |
| app/ai/{inference, preprocessing, contract} | ✅ |
| app/ai/**model**/ | ⚠️ not present: model weights live in `models/`, the runtime wrapper in `app/ai/inference/backend.py` |
| app/observation, storage, ui | ✅ |
| data/{train, evaluation, negative, metadata} | ✅ layout (+ `val`, `negative_val`, explained in `docs/dataset.md`); contents empty |
| models/ | ✅ (empty until you train) |
| tests/{ai, observation, storage} | ✅ (+ data, ui, ui_browser) |
| scripts/{evaluate, benchmark, export_model}.py | ✅ |
| .github/workflows | ✅ |

## The shortest path to ✅ everywhere that is still ❌
1. `docs/data_collection.md`: `probe` → `plan` → `select` → `download` → add your own non-plant photos and field photos → QC with contact sheets → `check_dataset.py` = 0 errors. *(criteria 11, 12)*
2. `scripts/train.py` → `scripts/export_model.py` (parity must pass) → `scripts/calibrate_threshold.py` → `scripts/evaluate.py` → `scripts/benchmark.py`. *(2, 3, 13–16)*
3. Walk to a real plant with the app. *(18)*
4. Choose a LICENSE.
